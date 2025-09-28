import duckdb
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os
import json
import random

DB = "data/transactions.duckdb"
MODEL_DIR = "data/models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_data():
    con = duckdb.connect(DB)
    df = con.execute("SELECT * FROM reporting_txns").df()
    con.close()
    if df.empty:
        raise SystemExit("reporting_txns empty — run pipeline.py first.")
    df['txn_date'] = pd.to_datetime(df['txn_date'])
    return df

def feature_engineer(df):
    # basic features
    df['hour'] = df['txn_date'].dt.hour
    df['dayofweek'] = df['txn_date'].dt.dayofweek
    # freq: transactions per customer in last 7 days (approx)
    df = df.sort_values(['customer_id','txn_date'])
    # simple rolling count by customer in past 7 days (approx via groupby)
    df['cust_txn_count_7d'] = df.groupby('customer_id')['txn_date'].transform(
        lambda x: x.rolling(50, min_periods=1).count()  # approximate sliding window
    ).fillna(0)
    # label encode simple categoricals and save encoders
    encoders = {}
    for col in ['txn_type','merchant_category','device']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = { 'classes': le.classes_.tolist() }
    # keep feature list
    features = ['amount','hour','dayofweek','cust_txn_count_7d','txn_type','merchant_category','device']
    return df, features, encoders

def create_synthetic_label(df):
    # Create a synthetic label:
    # suspicious if amount > 100000 OR random injected bursts (simulate fraud)
    df['label'] = (df['amount'] > 100000).astype(int)
    # inject a small fraction of additional suspicious rows (2% of rows)
    n = len(df)
    inject_n = max(1, int(0.02 * n))
    idx = np.random.choice(df.index, size=inject_n, replace=False)
    df.loc[idx, 'label'] = 1
    return df

def train_models(df, features):
    X = df[features].fillna(0).values
    # Unsupervised IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=RANDOM_SEED)
    iso.fit(X)
    iso_scores = -iso.decision_function(X)  # higher => more anomalous
    df['iso_score'] = iso_scores

    # Add iso_score into features for supervised model
    feat_with_iso = features + ['iso_score']
    X2 = df[feat_with_iso].fillna(0).values
    y = df['label'].values

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=RANDOM_SEED, n_jobs=-1)
    rf.fit(X_train, y_train)

    # eval
    y_pred = rf.predict(X_test)
    try:
        y_proba = rf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None

    print("Classification report on test set:")
    print(classification_report(y_test, y_pred))
    if auc is not None:
        print("ROC AUC:", auc)

    # save models and metadata
    joblib.dump(iso, os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_classifier.joblib"))
    meta = {
        "features": features,
        "feat_with_iso": feat_with_iso
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f)
    print("Saved models and metadata to", MODEL_DIR)

def main():
    df = load_data()
    df, features, encoders = feature_engineer(df)
    df = create_synthetic_label(df)
    # optionally save encoders
    joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.joblib"))
    train_models(df, features)

if __name__ == "__main__":
    main()
