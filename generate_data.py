
import csv
import random
from datetime import datetime, timedelta
from faker import Faker
import argparse
import os

fake = Faker()

parser = argparse.ArgumentParser()
parser.add_argument("--rows", type=int, default=2000, help="Number of transactions to generate")
parser.add_argument("--out", type=str, default="data/sample_txns.csv", help="Output CSV path")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

txn_types = ["debit", "credit"]
merchant_categories = ["Grocery", "Utilities", "Travel", "Entertainment", "Dining", "Salary", "Shopping", "Healthcare"]

with open(args.out, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["txn_id","customer_id","account_id","txn_date","amount","txn_type","merchant_category","location","device"])
    start = datetime.now() - timedelta(days=90)
    for i in range(args.rows):
        txn_id = f"TXN{100000+i}"
        customer_id = random.randint(1000, 1999)
        account_id = "AC" + str(random.randint(100000, 199999))
        txn_dt = start + timedelta(seconds=random.randint(0, 90*24*3600))
        amt = round(random.uniform(10, 250000), 2)
        ttype = random.choices(txn_types, weights=[0.85,0.15])[0]
        cat = random.choice(merchant_categories)
        loc = fake.city()
        device = random.choice(["mobile","web","pos"])
        writer.writerow([txn_id, customer_id, account_id, txn_dt.strftime("%Y-%m-%d %H:%M:%S"), amt, ttype, cat, loc, device])

print(f"Generated {args.rows} transactions to {args.out}")
