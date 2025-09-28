import duckdb
import pandas as pd
import numpy as np
import joblib
import os
import json
import shap

DB = "data/transactions.duckdb"
MODEL_DIR = "data/models"

def get_models():
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest.joblib"))
    rf = joblib.load(os.path.join(MODEL_DIR, "rf_classifier.joblib"))
    meta = json.load(open(os.path.join(MODEL_DIR, "model_meta.json")))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    return iso, rf, meta, encoders

def featurize(df, features, encoders):
    df['txn_date'] = pd.to_datetime(df['txn_date'])
    df['hour'] = df['txn_date'].dt.hour
    df['dayofweek'] = df['txn_date'].dt.dayofweek
    df = df.sort_values(['customer_id','txn_date'])
    df['cust_txn_count_7d'] = df.groupby('customer_id')['txn_date'].transform(
        lambda x: x.rolling(50, min_periods=1).count()
    ).fillna(0)
    # apply encoder mapping if present
    for col in ['txn_type','merchant_category','device']:
        if col in encoders:
            classes = encoders[col]['classes']
            mapping = {c: i for i, c in enumerate(classes)}
            df[col] = df[col].astype(str).map(lambda v: mapping.get(v, -1))
    return df

def _to_float_safe(x):
    """
    Convert x (scalar or small numpy array) to Python float safely.
    If x has multiple elements, take the first element.
    """
    try:
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.item())
        else:
            # flatten and take first element as fallback
            return float(arr.ravel()[0])
    except Exception:
        # last-resort: try Python float conversion
        return float(x)

def compute_and_write():
    con = duckdb.connect(DB)
    df = con.execute("SELECT * FROM reporting_txns").df()
    if df.empty:
        raise SystemExit("reporting_txns is empty; run pipeline.py first.")
    iso, rf, meta, encoders = get_models()
    features = meta['features']
    feat_with_iso = meta['feat_with_iso']

    df = featurize(df, features, encoders)
    X = df[features].fillna(0).values
    iso_scores = -iso.decision_function(X)
    df['iso_score'] = iso_scores

    X2 = df[feat_with_iso].fillna(0).values
    try:
        probs = rf.predict_proba(X2)[:,1]
    except Exception:
        probs = rf.predict(X2)
    df['ml_score'] = probs
    df['ml_flag'] = (df['ml_score'] >= 0.5).astype(int)

    # SHAP explainability
    explainer = shap.TreeExplainer(rf)
    shap_vals = explainer.shap_values(X2)
    shap_pos = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    top_feats = []
    for i in range(len(df)):
        row_shap = shap_pos[i]
        # Get top 3 indices as plain ints
        idxs = np.argsort(-np.abs(row_shap))[:3]
        idxs = [int(x) for x in np.ravel(idxs)]
        feats = []
        for j in idxs:
            # safe name and value extraction
            feat_name = feat_with_iso[j]
            shap_val = _to_float_safe(row_shap[j])
            feats.append((feat_name, shap_val))
        top_feats.append(feats)

    df['explain_top'] = [json.dumps(x) for x in top_feats]

    # write ml_results table
    write_df = df[['txn_id','customer_id','account_id','txn_date','amount',
                   'txn_type','merchant_category','device','ml_score','ml_flag','explain_top']]
    con.register("tmp_df", write_df)
    con.execute("DROP TABLE IF EXISTS ml_results")
    con.execute("CREATE TABLE ml_results AS SELECT * FROM tmp_df")
    con.close()
    print("Wrote ml_results table with", len(write_df), "rows.")

if __name__ == "__main__":
    compute_and_write()
