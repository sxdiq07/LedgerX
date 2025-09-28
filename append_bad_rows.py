# append_bad_rows.py
import csv
from datetime import datetime
path = "data/sample_txns.csv"

bad_rows = [
    # missing account_id
    ["TXN_BAD_1", 2001, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 1500.0, "debit", "Grocery", "CityX", "mobile"],
    # non-positive amount
    ["TXN_BAD_2", 2002, "AC000001", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), -50.0, "debit", "Utilities", "CityY", "web"],
    # bad date (empty)
    ["TXN_BAD_3", 2003, "AC000002", "", 500.0, "credit", "Salary", "CityZ", "pos"],
    # duplicate txn_id (duplicate of an existing id - maybe TXN10000 exists)
    ["TXN10000", 1000, "AC100000", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 100.0, "debit", "Shopping", "City1", "web"],
    # another missing account_id
    ["TXN_BAD_4", 2004, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 250.0, "debit", "Dining", "CityA", "mobile"],
    # amount exactly zero
    ["TXN_BAD_5", 2005, "AC000005", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0.0, "credit", "Travel", "CityB", "pos"],
]

# Append rows
with open(path, "a", newline="") as f:
    writer = csv.writer(f)
    for r in bad_rows:
        writer.writerow(r)

print(f"Appended {len(bad_rows)} bad rows to {path}")
