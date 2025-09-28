import duckdb
import pandas as pd
import os

DB_PATH = "data/transactions.duckdb"
CSV_PATH = "data/sample_txns.csv"

def connect_db(path=DB_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return duckdb.connect(path)

def ingest(conn, csv_path=CSV_PATH):
    print("Ingesting CSV into staging_txns...")
    df = pd.read_csv(csv_path, parse_dates=["txn_date"])
    conn.register("pandas_df", df)
    conn.execute("DROP TABLE IF EXISTS staging_txns")
    conn.execute("CREATE TABLE staging_txns AS SELECT * FROM pandas_df")
    print(f"Staged {len(df)} rows.")

def validate_and_transform(conn):
    print("Validating data and building reporting table...")
    conn.execute("DROP TABLE IF EXISTS validation_failures")
    conn.execute("""
    CREATE TABLE validation_failures AS
    SELECT txn_id,
           customer_id,
           account_id,
           txn_date,
           amount,
           CASE
             WHEN account_id IS NULL OR account_id = '' THEN 'missing_account'
             WHEN amount <= 0 THEN 'non_positive_amount'
             WHEN txn_date IS NULL THEN 'bad_date'
             ELSE NULL
           END AS failure_reason
    FROM staging_txns
    WHERE account_id IS NULL OR account_id = '' OR amount <= 0 OR txn_date IS NULL
    """)
    conn.execute("DROP TABLE IF EXISTS reporting_txns")
    conn.execute("""
    CREATE TABLE reporting_txns AS
    WITH dedup AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY txn_id ORDER BY txn_date DESC) AS rn
      FROM staging_txns
    )
    SELECT txn_id, customer_id, account_id, txn_date, amount, txn_type, merchant_category, location, device
    FROM dedup
    WHERE rn = 1
      AND txn_id NOT IN (SELECT txn_id FROM validation_failures)
    """)
    print("Reporting table created.")

    conn.execute("DROP VIEW IF EXISTS v_daily_summary")
    conn.execute("""
    CREATE VIEW v_daily_summary AS
    SELECT DATE(txn_date) AS day,
           COUNT(*) AS txn_count,
           SUM(amount) AS total_amount,
           AVG(amount) AS avg_amount
    FROM reporting_txns
    GROUP BY DATE(txn_date)
    ORDER BY day DESC
    """)
    print("Summary view created.")

def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found at {CSV_PATH}. Run generate_data.py first.")
    conn = connect_db()
    ingest(conn)
    validate_and_transform(conn)
    conn.close()
    print("Pipeline finished. DB at:", DB_PATH)

if __name__ == '__main__':
    main()
