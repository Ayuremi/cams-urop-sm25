# WIP GenAI Model for Risk Prediction

Data sourced from [cvelistV5](https://github.com/CVEProject/cvelistV5?tab=readme-ov-file). 
Follow download instructions [here](https://github.com/CVEProject/cvelistV5?tab=readme-ov-file#how-to-download-the-cve-list).   
The programs will expect CVE file paths to be structured as `./cves/<YEAR>/<THOUSAND>/CVE-*.json`

`data-compiler.py` was run to create `cves_final.parquet`, the data used to train and validate `risk_model.pkl` in `model.py`.  
You can run `data-compiler.py` to create `cves_final.parquet` and run `model.py` to create `risk_model.pkl`.

--- 

**`genai.py` hosts the LLM, including classes for Chroma database, NER (Named Entity Recognition) model, and risk model.**
- This model utilized LangChain and HuggingFace models. Please download any required dependencies using `pip install <DEPENDENCY>`. These include, but may not be limited to:

  ```
  pip install polars
  pip install xgboost
  pip install scikit-learn
  pip install matplotlib
  pip install langchain-core
  pip install langchain-chroma
  pip install langchain-groq
  pip install langchain-huggingface
  pip install transformers
  pip install rapidfuzz
  ```
- Note that if you continue to use `groq`, request an API key and set API key in environment variable  
  ```
  export GROQ_API_KEY=<API-KEY-HERE>
  ```
- Running `genai.py` for the first time will create Chroma database and embeddings at `./chroma_db`

---

General process/idea:

```
User query -> NER -> risk predicition tool (with fuzzy matching and feature extraction) -> LLM prompt engineering with risk and documents -> response
                  -> Chroma database document similarity search                         ->
```

To complete model:
- Create tool functions for LLM to call (NER, risk assessment, document retrieval for semantic search and RAG)
- Integrate with LLM agent with prompt engineering and tool use

Possible improvements to make:
- Auto CVE data download from cvelistv5 (or better source)
- Up-to-date unique vendor/product list (currently static file `unique_products.csv` generated from `cves_final.parquet` for fuzzy matching)
- Improve/train NER model for better entity extraction
- Better risk prediction model (more features, different model)
- Better total_vendor_vulns calculation (currently estimate count)

