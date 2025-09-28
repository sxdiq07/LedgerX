import duckdb
import pandas as pd
import sys

DB = "data/transactions.duckdb"
OUT = "data/ml_results.csv"

try:
    con = duckdb.connect(DB)
    df = con.execute("SELECT * FROM ml_results").df()
    con.close()
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} with {len(df)} rows")
except Exception as e:
    print("ERROR exporting ml_results:", e)
    sys.exit(1)
