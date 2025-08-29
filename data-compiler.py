import polars as pl
import datetime as dt
import os
import json

DATE = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)  # Set the date to filter vulnerabilities
TODAY = dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc)

def extract_cve(path, save = False) -> pl.DataFrame:
    ''' Extract CVE data from JSON files in the specified directory.'''
    schema = {
        "vendor": pl.String,
        "product": pl.String,
        "cve": pl.String,
        "date_published": pl.String,
    }
    
    def extract_info(cve: dict) -> pl.DataFrame:
        output = pl.DataFrame(schema=schema)
        
        id = cve.get("cveMetadata", {}).get("cveId", None)
        date_published = cve.get("cveMetadata", {}).get("datePublished", None)
        affected = cve.get("containers", {}).get("cna", {}).get("affected", [])
        for item in affected:
            
            new = pl.DataFrame({
                "vendor": item.get("vendor", None),
                "product": item.get("product", None),
                "cve": id,
                "date_published": date_published,
            })
            output = pl.concat([output, new], how="vertical")
        return output
        
    def parse_all_cve_json(cves_root):
        rows = []
        for root, dirs, files in os.walk(cves_root):
            print(f"Scanning directory: {root}, found {len(files)} files.")
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            cve = json.load(f)
                            new = extract_info(cve)
                            rows.append(new)
                    except Exception as e:
                        print(f"Error parsing {file_path}: {e}")
        if rows:
            return pl.concat(rows, how="vertical")
        else:
            return pl.DataFrame(schema=schema)
    
    all_cves = parse_all_cve_json(path).unique(subset=['vendor', 'product', 'cve']).drop_nulls()
    all_cves = all_cves.remove((pl.col('vendor') == 'n/a') | (pl.col('product') == 'n/a'))
    if save: all_cves.write_parquet('pyvul_data.parquet')
    return all_cves


def extract_cvss(save = False) -> pl.DataFrame:
    ''' Extract CVSS scores using CIRCL Hugging Face database.'''
    # Login using e.g. `huggingface-cli login` to access this dataset
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    scores_train = pl.read_parquet("hf://datasets/CIRCL/vulnerability-scores/" + splits["train"])
    scores_test = pl.read_parquet("hf://datasets/CIRCL/vulnerability-scores/" + splits["test"])
    scores = scores_train.vstack(scores_test)
    scores = scores.drop(['title', 'description', 'cpes'])
    scores = scores.with_columns(pl.coalesce('cvss_v4_0', 'cvss_v3_1', 'cvss_v3_0', 'cvss_v2_0').alias('cvss'))
    scores = scores.drop(['cvss_v4_0', 'cvss_v3_1', 'cvss_v3_0', 'cvss_v2_0'])
    if save: scores.write_parquet('cvss.parquet')
    return scores
    

def aggregate_cves(save = False) -> pl.DataFrame:
    ''' Aggregate CVE data with CVSS scores.'''
    cves = extract_cve('./cves/2024/', save=False)
    cvss = extract_cvss(save=False)
    df = cves.join(cvss, left_on='cve', right_on='id', how='left')
    
    if save: df.write_parquet('pyvul_scores.parquet')
    
    df = pl.read_parquet('pyvul_scores.parquet')  # reload dataset
    df = df.with_columns((TODAY - pl.col('date_published').str.to_datetime(strict=False)).dt.total_days().alias('days_since_published'))
    df = df.filter(pl.col('days_since_published') > 0) # filter out future dates
    df = df.sort('days_since_published', descending=False)  # sort by date published

    df = df.group_by(['vendor', 'product']).agg([
        pl.count('cve').alias('num_vulns'),
        pl.mean('cvss').alias('avg_cvss'),
        pl.max('cvss').alias('max_cvss'),
        pl.min('cvss').alias('min_cvss'),
        pl.col('days_since_published').min().alias('min_days_since_published'),
        pl.col('days_since_published').diff().mean().alias('avg_days_between_vulns'),
        ]
    )
    df = df.with_columns(pl.col('avg_days_between_vulns').fill_null(0))  # fill nulls with 0
    df = df.with_columns(pl.when(pl.col("avg_days_between_vulns") == 0).then(1).otherwise(0).alias("is_time_null"))
    df = df.sort('num_vulns', descending=True)
    print(df.sort('num_vulns', descending=True).head(100))
    if save: df.write_parquet('pyvul_scores_v2.parquet')  # save with new aggregations
    
    return df
    

def add_pop_opensrc_vendor(save = False) -> pl.DataFrame:
    ''' Add popularity, open source detection, and vendor frequency to the aggregated CVE data.'''
    df = pl.read_parquet('pyvul_scores_v2.parquet') if os.path.exists('pyvul_scores_v2.parquet') else aggregate_cves(save=True)

    # Define open source vendor/product keywords
    open_source_keywords = [
        "linux", "apache", "mozilla", "canonical", "red hat", "debian", "kde", "gnome",
        "alpine", "openssl", "eclipse", "python", "node.js", "kubernetes", "docker",
        "libreoffice", "postgresql", "sqlite", "mariadb", "nginx"
    ]

    # Define product popularity heuristics
    def estimate_popularity_static(vendor, product, num_vulns) -> float:
        combined = f"{vendor} {product}".lower()
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
    def check_open_source(vendor: str, product: str) -> int:
        combined = f"{vendor} {product}".lower()
        return any(keyword in combined for keyword in open_source_keywords)

    # Apply functions using Polars expressions
    df = df.with_columns([
        pl.struct(["vendor", "product"]).map_elements(
            lambda x: check_open_source(x["vendor"], x["product"]),
            return_dtype=pl.Int8
        ).alias("is_open_source"),

        pl.struct(["vendor", "product", "num_vulns"]).map_elements(
            lambda x: estimate_popularity_static(x["vendor"], x["product"], x["num_vulns"]),
            return_dtype=pl.Float64
        ).alias("popularity_score"),
        
        pl.col('num_vulns').sum().over('vendor').alias('vendor_total_vulns')
    ])

    # Save to enhanced CSV
    if save: df.write_parquet("pyvul_scores_v3.parquet")
    

def has_future_vuln(days, save = False) -> pl.DataFrame:
    ''' Check if there are future vulnerabilities within the next `days` days.'''
    df = pl.read_parquet('pyvul_scores_v3.parquet') if os.path.exists('pyvul_scores_v3.parquet') else add_pop_opensrc_vendor(save=True)
    
    future = pl.read_parquet('pyvul_future.parquet') if os.path.exists('pyvul_future.parquet') else extract_cve('./cves/2025/')
    next3M = (future.filter(pl.col('date_published').str.to_datetime(strict=False) > TODAY)
            .filter(pl.col('date_published').str.to_datetime(strict=False) <= (TODAY + dt.timedelta(days=days)))
            .unique(subset=['vendor', 'product'])
            .with_columns(pl.lit(1).alias('has_future_vuln'))
            .select(['vendor', 'product', 'has_future_vuln'])
            )
    df = df.join(next3M, on=['vendor', 'product'], how='left')
    df = df.with_columns(pl.col('has_future_vuln').fill_null(0).cast(pl.Int8))
    
    if save: df.write_parquet('pyvul_scores_v4.parquet')
    return df
    
    
if __name__ == "__main__":
    df = has_future_vuln(90, save=True)
    df.write_parquet('cves_final.parquet')
    