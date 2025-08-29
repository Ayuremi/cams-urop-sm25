from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from langchain_core.tools import tool
from rapidfuzz import process, fuzz
import polars as pl
import datetime as dt

''' To do:
- Create tool functions for LLM to call (NER, risk assessment,
  document retrieval for semantic search and RAG)
- Integrate with LLM agent with prompt engineering and tool use

Possible improvements to make:
- Auto CVE data download from cvelistv5 (or better source)
- Up-to-date unique vendor/product list (currently static file for fuzzy matching)
- Improve NER model for better entity extraction
- Better risk prediction model (more features, different model)
- Better total_vendor_vulns calculation (currently estimate count)
'''

TODAY = dt.datetime(2025, 8, 29, tzinfo=dt.timezone.utc)

class ChromaDatabase:
    def __init__(self, persist_directory="./chroma_db", collection_name="cve_collection", years=["2024", "2025"]):
        # years should be abt 1 year period
        # using 2024 and 2025 here to capture recent CVEs
        # w/o need for explicit filter logic (can be added later if needed)
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.years = years
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            show_progress=True,
            )
        
        # Load or create the Chroma vector store
        if self._chroma_exists():
            self.db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
        else:
            self.db = self._create_persist_vector_store()
            
        self.unique_ven_prod = self._get_unique_vendor_product_lists()      
        self.df = self._get_cve_dataframe()
        
    
    def _chroma_exists(self):
        import os 
        return os.path.exists(self.persist_directory) and len(os.listdir(self.persist_directory)) > 0
    
    
    def _create_persist_vector_store(self):
        # assumes CVE JSON files are stored in cves/<year>/*.json from cvelistv5
        from langchain_community.document_loaders import DirectoryLoader, JSONLoader
        
        jq_schema = r'''
        # select(.cveMetadata.state == "PUBLISHED") |
        {
        id: (.cveMetadata.cveId // ""),
        state: (.cveMetadata.state // ""),
        title: (.containers.cna.title // ""),
        description: (
            [.containers.cna.descriptions[]? 
                | select(.lang=="en") 
                | .value // ""]
            | map(select(. != null and . != ""))
            | join(" ")
        ),
        date_published: (.cveMetadata.datePublished // ""),
        vendor: (
            [.containers.cna.affected[]?.vendor // ""]
            | map(select(. != null and . != ""))
            | join("; ")
        ),
        product: (
            [.containers.cna.affected[]?.product // ""]
            | map(select(. != null and . != ""))
            | join("; ")
        ),
        versions: (
            [.containers.cna.affected[]?.versions[]?.version // ""]
            | map(select(. != null and . != ""))
            | join("; ")
        ),
        cwe: (
            [.containers.cna.problemTypes[]?.descriptions[]? 
                | select(.lang=="en") 
                | .description // ""]
            | map(select(. != null and . != ""))
            | join(", ")
        ),
        cvss_v3_baseScore: (.containers.cna.metrics[]?.cvssV3_1.baseScore // ""),
        cvss_v3_severity: (.containers.cna.metrics[]?.cvssV3_1.baseSeverity // ""),
        references: (
            [.containers.cna.references[]?.url // ""]
            | map(select(. != null and . != ""))
            | join(", ")
        ),
        }
        # Create a single text blob for embeddings
        | .embedding_text = (
            [
                (.title // ""),
                (.vendor // ""),
                (.product // ""),
                "Versions: " + (.versions // "" | .[:25])
            ]
            | map(select(. != null and . != ""))
            | join(" | ")
        )
        '''


        def metadata_func(record, _):
            metadata = {k: v for k, v in record.items() if k != "embedding_text"}
            return metadata
            
            
        # Import local CVE data
        loaders = [
            DirectoryLoader(
                path=f"cves/{year}",
                glob="**/*.json",
                loader_cls=JSONLoader,
                loader_kwargs={
                    "jq_schema": jq_schema,
                    "metadata_func": metadata_func,
                    "content_key": "embedding_text",
                    "text_content": False,
                    },
                show_progress=True,
                )
            for year in self.years
        ]
        cves = []
        for loader in loaders:
            for doc in loader.load():
                # Only keep PUBLISHED CVEs
                if doc.metadata.get("state")=="PUBLISHED":
                    cves.append(doc)

        # Create vector store using Chroma
        return Chroma.from_documents(
            documents=cves,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
        )
    
    
    def _get_unique_vendor_product_lists(self):
        # Load unique vendor/product list from static file
        # This can be improved by auto-updating from CVE data periodically
        # or extracting from chroma DB 
        df = pl.read_csv('unique_products.csv')
        vendors = df['vendor'].unique().to_list()
        products = df['product'].unique().to_list()
        return {'vendors': vendors, 'products': products}
    
    
    def _get_cve_dataframe(self):
        df = pl.DataFrame(self.db._collection.get()['metadatas'])
        df = df.with_columns([
            pl.col('vendor').str.split('; ').alias('vendor'),
            pl.col('product').str.split('; ').alias('product'),
        ])  
        df = df.explode('vendor').explode('product')
        df = df.remove((pl.col('state') != 'PUBLISHED'))
        return df
    
    
    def query(self, question, k=20):
        results = self.db.similarity_search(question, k=k)
        return results


class NERExtractor:
    def __init__(self, model_id="danitamayo/bert-cybersecurity-NER"):
        self.tokenizer_ner = AutoTokenizer.from_pretrained(model_id) # hugging face model
        self.ner_model = AutoModelForTokenClassification.from_pretrained(model_id,ignore_mismatched_sizes=True)
        self.ner = pipeline("ner",
                    model=self.ner_model,
                    tokenizer=self.tokenizer_ner,
                    aggregation_strategy="max")
    
    def extract(self, query):
        return self.ner(query)
    
    
class RiskModel:
    def __init__(self):
        import joblib
        self.model = joblib.load('risk_model.pkl')


    def fuzzy_match(self, query, choices, limit=10, score_cutoff=75):
        matches = process.extract(
            query,
            choices,
            scorer=fuzz.partial_token_sort_ratio,
            limit=limit,
            score_cutoff=score_cutoff
        )
        return [match[0] for match in matches]


    def filter_cve_with_fuzzy(self, cves, query, cdb, limit=10, score_cutoff=75):
        # Fuzzy match vendors and products from query
        ven_matches = self.fuzzy_match(query, cdb.unique_ven_prod['vendors'], limit=limit, score_cutoff=score_cutoff)
        prod_matches = self.fuzzy_match(query, cdb.unique_ven_prod['products'], limit=limit, score_cutoff=score_cutoff)
        
        # Filter CVEs based on matched vendors/products
        # Assumes cves is exploded on vendor/product columns
        filtered = cves.filter(
            (pl.col('vendor').is_in(ven_matches)) & 
            (pl.col('product').is_in(prod_matches))
        )
        return filtered



    def create_features(self, df, cdb_df, ner_text, today= TODAY):
        df = df['vendor', 'product', 'date_published', 'cvss_v3_baseScore']
        df = df.with_columns((today - pl.col('date_published').str.to_datetime(strict=False)).dt.total_days().alias('days_since_published'))
        df = df.filter(pl.col('days_since_published') > 0) # filter out future dates
        df = df.sort('days_since_published', descending=False)  # sort by date published
        df = df.with_columns(pl.col('cvss_v3_baseScore').cast(pl.Float64))
        
        df = df.with_columns([
            pl.len().alias('num_vulns'),
            pl.mean('cvss_v3_baseScore').alias('avg_cvss'),
            pl.max('cvss_v3_baseScore').alias('max_cvss'),
            pl.min('cvss_v3_baseScore').alias('min_cvss'),
            pl.col('days_since_published').min().alias('min_days_since_published'),
            pl.col('days_since_published').diff().mean().alias('avg_days_between_vulns'),
            ]
        )
        df = df.with_columns(pl.col('avg_days_between_vulns').fill_null(0))  # fill nulls with 0
        df = df.with_columns(pl.when(pl.col("avg_days_between_vulns") == 0).then(1).otherwise(0).alias("is_time_null"))
        
        # count total vulns for vendor
        total_vulns = cdb_df.filter(pl.col('vendor').is_in(df['vendor'].unique().to_list())).height
        df = df.with_columns(pl.lit(total_vulns // df.height).alias('total_vendor_vulns'))
        
        # add open source and popularity features
        open_source_keywords = [
            "linux", "apache", "mozilla", "canonical", "red hat", "debian", "kde", "gnome",
            "alpine", "openssl", "eclipse", "python", "node.js", "kubernetes", "docker",
            "libreoffice", "postgresql", "sqlite", "mariadb", "nginx"
        ]

        # Define product popularity heuristics
        def estimate_popularity_static(ner_text, num_vulns) -> float:
            combined = ner_text.lower()
            if any(x in combined for x in ['windows', 'linux', 'android', 'ios', 'mac', 'chrome', 'office']):
                return 0.95
            elif any(x in combined for x in ['apache', 'nginx', 'mysql', 'postgres', 'docker', 'kubernetes']):
                return 0.85
            elif any(x in combined for x in ['python', 'node', 'java', 'openssl']):
                return 0.75
            elif num_vulns > 30:
                return min(0.7, 0.1 + (num_vulns / df['num_vulns'].max())**0.5)
            else:
                return 0.1

        # Open source detection function
        def check_open_source(ner_text) -> int:
            combined = ner_text.lower()
            return any(keyword in combined for keyword in open_source_keywords)

        # Apply functions using Polars expressions
        df = df.with_columns([
            pl.lit(estimate_popularity_static(ner_text, df['num_vulns'][0])).alias('popularity_score'),
            pl.lit(check_open_source(ner_text), int).alias('is_open_source')
            ]
        )
        
        return df
    
    
    def predict_risk(self, cdb, ner_results):
        df = cdb.df
        ner_text = " ".join([dic["word"] for dic in ner_results])
        
        # feature engineering
        filtered_cves = self.filter_cve_with_fuzzy(df, ner_text, cdb, limit=10, score_cutoff=75)
        features_df = self.create_features(filtered_cves, df, ner_text)
        
        # load model   
        risk_score = self.model.predict(features_df.drop(['vendor', 'product', 'date_published', 'cvss_v3_baseScore', 'days_since_published'])[0:1])
        print("Predicted risk score:", risk_score[0])
        return risk_score[0]


if __name__ == "__main__":
    # Initialize Groq LLM
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.7
    )

    cdb = ChromaDatabase()

    ner = NERExtractor()

    query = "What are the known vulnerabilities for Microsoft Exchange Server?"
    ner_results = ner.extract(query)
    risk_model = RiskModel()
    risk_score = risk_model.predict_risk(cdb, ner_results)


## WIP below: integrate with LLM agent and tool use

# def ner_extractor_tool(query: str) -> list:
#     '''Extract named entities (vendor, product) from the input text using a pre-trained NER model.
#     Args:
#         query (str): Input text containing potential vendor and product names.
#     '''
#     return ner.extract(query)

# def risk_assessor_tool(ner_result: list[dict]) -> float:
#     ''' Assess the risk score based on extracted NER results and known vulnerabilities.
#     Args:
#         ner_result (list[dict]): List of dictionaries containing extracted entities
#     '''
#     risk_score = risk_model.predict_risk(cdb, ner_result)
#     return risk_score

# tools = [ner_extractor_tool, risk_assessor_tool]
# llm_with_tools = llm.bind_tools(tools)

# from langchain_core.messages import HumanMessage

# query = "What is the risk for Adobe Animate?"

# messages = [HumanMessage(query)]

# ai_msg = llm_with_tools.invoke(messages)

# print(ai_msg.tool_calls)

# messages.append(ai_msg)

    
