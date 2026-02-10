
import streamlit as st
import pandas as pd
import sys
import os

# Add src to python path to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_engine.model import BenchmarkAI

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Transaction Insight Platform",
    page_icon="💸",
    layout="wide"
)

# --- Title and Header ---
st.title("💸 AI-Driven Intelligent Transaction Analysis Platform")
st.markdown("""
This platform uses **Artificial Intelligence** to analyze transaction patterns. 
It detects high-risk anomalies, scores transaction safety, and provides insights for financial monitoring.
""")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
data_source = st.sidebar.radio("Select Data Source", ["Generate Synthetic Data", "Upload CSV"])

# --- AI Engine Loading ---
@st.cache_resource
def load_ai_engine():
    """Load the AI model once and cache it."""
    return BenchmarkAI()

try:
    ai = load_ai_engine()
except Exception as e:
    st.error(f"Failed to load AI Engine. Error: {e}")
    st.stop()

# --- Main Logic ---
df = None

if data_source == "Generate Synthetic Data":
    st.sidebar.subheader("Simulation Settings")
    duration = st.sidebar.slider("Number of Records", 100, 5000, 1000)
    
    if st.sidebar.button("Generate Transactions", type="primary"):
        with st.spinner("Generating transaction data..."):
            try:
                import random
                from datetime import datetime, timedelta
                data = []
                for i in range(duration):
                    data.append({
                        "transaction_id": f"TXN{1000+i}",
                        "amount": round(random.uniform(10, 5000), 2),
                        "tx_hour": random.randint(0, 23),
                        "account_age_days": random.randint(1, 1000),
                        "transaction_date": (datetime.now() - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
                    })
                df = pd.DataFrame(data)
                st.session_state["data"] = df
                st.toast(f"Generated {len(df)} transactions!", icon="✅")
            except Exception as e:
                st.error(f"Generation failed: {e}")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type="csv")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["data"] = df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# Load data from session state if available
if "data" in st.session_state:
    df = st.session_state["data"]

# --- Display & Analysis ---
if df is not None:
    st.divider()
    
    # Top Section: Data Preview
    st.subheader("📊 Raw Transaction Metrics")
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze with AI", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("🤖 AI is analyzing transactions..."):
            try:
                # Prepare data for AI
                metrics = df.to_dict(orient="records")
                
                # Validation for transaction columns
                required_cols = ["amount", "tx_hour", "account_age_days"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                     raise ValueError(f"Required transaction columns missing: {', '.join(missing)}")
                
                results = ai.detect_anomalies(metrics)
                
                # Merge results back to DF
                results_df = pd.DataFrame(results)
                
                # Rename detection column to avoid collision
                if "is_anomaly" in results_df.columns:
                    results_df.rename(columns={"is_anomaly": "detected_anomaly"}, inplace=True)
                
                final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                
                st.session_state["results"] = final_df
                st.toast("Analysis Complete!", icon="🎉")
            except ValueError as ve:
                st.error(f"❌ **Invalid Data Format:** {ve}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# --- Results Presentation ---
if "results" in st.session_state:
    res = st.session_state["results"]
    st.divider()
    
    # KPI Metrics
    try:
        avg_score = res["performance_score"].mean()
        anomalies_count = res["detected_anomaly"].sum() 
        total_amount = res["amount"].sum()
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Average Risk Score", f"{avg_score:.1f} / 100", delta_color="normal")
        kpi2.metric("Flagged Anomalies", f"{anomalies_count}", delta_color="inverse")
        kpi3.metric("Total Transaction Volume", f"${total_amount:,.2f}")
    except KeyError as e:
        st.error(f"Missing expected columns in results: {e}")

    # Charts
    st.subheader("📈 Transaction Value Trends")
    chart_data = res[["amount", "performance_score"]]
    st.line_chart(chart_data)

    # Anomaly Drill-down
    if anomalies_count > 0:
        st.subheader("🚨 Identified High-Risk Transactions")
        anomaly_df = res[res["detected_anomaly"] == True]
        
        for idx, row in anomaly_df.iterrows():
            with st.status(f"Flagged Transaction {row.get('transaction_id', idx)} (Score: {row['performance_score']})", state="error"):
                st.write(f"**AI Insight:** {row['recommendation']}")
                st.json({
                    "Amount": f"${row['amount']:,.2f}",
                    "Hour": f"{row['tx_hour']}:00",
                    "Account Age": f"{row['account_age_days']} days",
                })
    else:
        st.success("✅ No suspicious patterns detected. Transactions appear normal.")

    # Full Data Table
    with st.expander("📂 View Full Risk Analysis Report"):
        st.dataframe(res, use_container_width=True)
