"""
app.py — LedgerX Streamlit Dashboard
- Sidebar filters
- KPIs header
- Interactive charts (Plotly)
- Validation failures + CSV download
- ML-flagged transactions
- Ad-hoc SQL
"""

import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import json
from datetime import datetime, timedelta

DB_PATH = "data/transactions.duckdb"

st.set_page_config(page_title="LedgerX", layout="wide", initial_sidebar_state="expanded")
st.title("LedgerX — Financial Compliance Analytics Dashboard")
st.caption("By Sadiq")

# -----------------------
# DB helpers
# -----------------------
def get_conn():
    return duckdb.connect(DB_PATH)

@st.cache_data(ttl=300)
def load_reporting_txns():
    con = get_conn()
    try:
        df = con.execute("SELECT * FROM reporting_txns").df()
    finally:
        con.close()
    df['txn_date'] = pd.to_datetime(df['txn_date'])
    return df

@st.cache_data(ttl=300)
def load_validation_failures():
    con = get_conn()
    try:
        df = con.execute("SELECT * FROM validation_failures").df()
    finally:
        con.close()
    if not df.empty:
        df['txn_date'] = pd.to_datetime(df['txn_date'], errors='coerce')
    return df

# -----------------------
# Load data
# -----------------------
df = load_reporting_txns()
df_fail = load_validation_failures()

if df.empty:
    st.warning("No reporting transactions found. Run `python pipeline.py` to create reporting_txns.")
    st.stop()

# -----------------------
# Sidebar - Filters
# -----------------------
st.sidebar.header("Filters")
min_date = df['txn_date'].min().date()
max_date = df['txn_date'].max().date()
date_range = st.sidebar.date_input("Date range", [max_date - timedelta(days=30), max_date], min_value=min_date, max_value=max_date)

txn_types = ["all"] + sorted(df['txn_type'].dropna().unique().tolist())
txn_type_sel = st.sidebar.selectbox("Transaction type", txn_types, index=0)

merchants = ["all"] + sorted(df['merchant_category'].dropna().unique().tolist())
merchant_sel = st.sidebar.multiselect("Merchant category (multi)", merchants if merchants else ["all"], default=["all"])

min_amount = st.sidebar.number_input("Minimum amount (₹)", min_value=0.0, value=0.0, step=100.0)
top_n = st.sidebar.slider("Top N customers", min_value=5, max_value=50, value=10)

customer_list = ["all"] + sorted(df['customer_id'].astype(str).unique().tolist())
customer_sel = st.sidebar.selectbox("Customer (optional)", customer_list, index=0)

# Apply filters
start_date, end_date = date_range[0], date_range[1]
mask = (df['txn_date'].dt.date >= start_date) & (df['txn_date'].dt.date <= end_date)
if txn_type_sel != "all":
    mask &= (df['txn_type'] == txn_type_sel)
if merchant_sel and "all" not in merchant_sel:
    mask &= df['merchant_category'].isin(merchant_sel)
if min_amount > 0:
    mask &= (df['amount'] >= float(min_amount))
if customer_sel != "all":
    mask &= (df['customer_id'].astype(str) == customer_sel)

df_f = df.loc[mask].copy()

# -----------------------
# Header KPIs
# -----------------------
col1, col2, col3, col4 = st.columns([2,2,2,1])
col1.metric("Transactions", f"{len(df_f):,}")
col2.metric("Total Volume (₹)", f"{df_f['amount'].sum():,.2f}")
col3.metric("Average Amount (₹)", f"{df_f['amount'].mean():,.2f}" if len(df_f)>0 else "0")
col4.metric("Suspicious (>₹100k)", f"{(df_f['amount']>100000).sum():,}")

st.markdown("---")

# -----------------------
# Charts row 1: time series + top customers
# -----------------------
c1, c2 = st.columns((2,1))
with c1:
    st.subheader("Daily Volume")
    daily = df_f.copy()
    daily['day'] = daily['txn_date'].dt.date
    daily_agg = daily.groupby('day')['amount'].sum().reset_index().sort_values('day')
    if daily_agg.empty:
        st.info("No daily data for selected filters.")
    else:
        fig = px.area(daily_agg, x='day', y='amount',
                      title="Daily transaction volume",
                      labels={'day':'Date','amount':'Amount (₹)'})
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0), height=400)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader(f"Top {top_n} Customers by Volume")
    top_cust = df_f.groupby('customer_id')['amount'].sum().reset_index().sort_values('amount', ascending=False).head(top_n)
    if top_cust.empty:
        st.info("No customers for selected filters.")
    else:
        fig2 = px.bar(top_cust, x='amount', y='customer_id', orientation='h',
                      title="Top customers", labels={'amount':'Amount (₹)','customer_id':'Customer ID'})
        fig2.update_layout(yaxis={'categoryorder':'total ascending'}, height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# -----------------------
# Charts row 2: distributions
# -----------------------
d1, d2 = st.columns(2)
with d1:
    st.subheader("Amount Distribution")
    if df_f.empty:
        st.info("No data")
    else:
        fig3 = px.histogram(df_f, x='amount', nbins=80,
                            title="Transaction amount distribution",
                            labels={'amount':'Amount (₹)'})
        fig3.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig3, use_container_width=True)

with d2:
    st.subheader("Transactions by Merchant Category")
    if df_f.empty:
        st.info("No data")
    else:
        cat = df_f['merchant_category'].value_counts().reset_index()
        cat.columns = ['merchant_category','count']
        fig4 = px.pie(cat, names='merchant_category', values='count', title='Merchant category share')
        fig4.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0))
        st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# -----------------------
# Table: raw transactions + export
# -----------------------
st.subheader("Filtered Transactions (preview)")
if df_f.empty:
    st.write("No transactions for these filters.")
else:
    st.dataframe(df_f.sort_values('txn_date', ascending=False).head(200))
    csv_buffer = df_f.to_csv(index=False).encode('utf-8')
    st.download_button("Download filtered CSV", csv_buffer, file_name="filtered_transactions.csv", mime="text/csv")

st.markdown("---")

# -----------------------
# Validation failures
# -----------------------
st.subheader("Validation Failures / Data Issues")
if df_fail.empty:
    st.info("No validation failures found.")
else:
    df_fail_f = df_fail.copy()
    df_fail_f['txn_date'] = pd.to_datetime(df_fail_f['txn_date'], errors='coerce')
    if not df_fail_f.empty:
        df_fail_f = df_fail_f[(df_fail_f['txn_date'].dt.date >= start_date) &
                              (df_fail_f['txn_date'].dt.date <= end_date)]
    st.dataframe(df_fail_f)
    if not df_fail_f.empty:
        st.download_button("Download failures CSV", df_fail_f.to_csv(index=False).encode('utf-8'), file_name="validation_failures.csv", mime="text/csv")

st.markdown("---")

# -----------------------
# ML-flagged Transactions
# -----------------------
st.subheader("ML-flagged Transactions (model output)")
try:
    con = get_conn()
    ml_df = con.execute("SELECT * FROM ml_results WHERE ml_flag=1 ORDER BY ml_score DESC LIMIT 200").df()
    con.close()
    if ml_df.empty:
        st.info("No ML results yet. Run model_inference.py to generate ml_results.")
    else:
        def explain_to_str(js):
            try:
                arr = json.loads(js)
                return "; ".join([f"{f}:{v:.3f}" for f,v in arr])
            except:
                return ""
        ml_df['explain'] = ml_df['explain_top'].apply(explain_to_str)
        st.dataframe(
            ml_df[['txn_id','customer_id','amount','ml_score','explain']].sort_values('ml_score', ascending=False).head(200)
        )
        st.download_button("Download ML flagged CSV", ml_df.to_csv(index=False).encode('utf-8'), file_name="ml_flagged.csv")
except Exception as e:
    st.info("ML results not found or error: " + str(e))

st.markdown("---")

# -----------------------
# Ad-hoc SQL
# -----------------------
st.subheader("Run ad-hoc SQL")
sql = st.text_area("Write a SQL query against the DuckDB database (tables: staging_txns, reporting_txns, validation_failures, v_daily_summary, ml_results)", height=180)
if st.button("Run SQL"):
    if not sql.strip():
        st.warning("Please enter a SQL query.")
    else:
        try:
            con = get_conn()
            res = con.execute(sql).df()
            con.close()
            st.dataframe(res)
            st.download_button("Download result CSV", res.to_csv(index=False).encode('utf-8'), file_name="sql_result.csv")
        except Exception as e:
            st.error(f"SQL error: {e}")

# -----------------------
# Sidebar Footer
# -----------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### ℹ️ About LedgerX
    **LedgerX** is a student-built, end-to-end demo of a compliance analytics system.  
    It ingests synthetic transaction data, validates it, and produces clear, interactive insights for compliance teams.  

    **Stack:** Python · DuckDB · Streamlit · Plotly  
    **Data:** 100% synthetic (Faker-generated)  
    """
)
