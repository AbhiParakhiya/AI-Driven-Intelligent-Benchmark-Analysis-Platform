
import streamlit as st
import pandas as pd
import sys
import os

# Add src to python path to import modules
# This allows importing ai_engine and data_gen directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_engine.model import BenchmarkAI

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Benchmark Platform",
    page_icon="🚀",
    layout="wide"
)

# --- Title and Header ---
st.title("🚀 AI-Driven Intelligent Benchmark Analysis Platform")
st.caption("v1.1 - Robust Validation Active")
st.markdown("""
This platform uses **Artificial Intelligence** to analyze system benchmark metrics. 
It detects anomalies, scores performance, and provides actionable recommendations.
""")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
data_source = st.sidebar.radio("Select Data Source", ["Generate Synthetic Data", "Upload CSV"])

if data_source == "Upload CSV":
    st.sidebar.info("""
    **Required CSV Columns:**
    - `cpu_usage` (%)
    - `memory_usage` (MB)
    - `disk_io` (MB/s)
    - `network_latency` (ms)
    - `throughput` (req/s)
    """)
    
    # Provide a sample template for download
    sample_data = {
        "cpu_usage": [45.2, 88.1, 32.5],
        "memory_usage": [3072, 7168, 2048],
        "disk_io": [120, 5, 150],
        "network_latency": [25, 450, 15],
        "throughput": [1200, 300, 1400]
    }
    sample_df = pd.DataFrame(sample_data)
    csv = sample_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="📥 Download Sample Template",
        data=csv,
        file_name="benchmark_template.csv",
        mime="text/csv",
        use_container_width=True
    )

# --- AI Engine Loading ---
@st.cache_resource
def load_ai_engine():
    """Load the AI model once and cache it."""
    return BenchmarkAI()

try:
    ai = load_ai_engine()
except Exception as e:
    st.error(f"Failed to load AI Engine. Make sure models are trained. Error: {e}")
    st.stop()

# --- Main Logic ---
df = None

if data_source == "Generate Synthetic Data":
    st.sidebar.subheader("Simulation Settings")
    duration = st.sidebar.slider("Duration (minutes)", 1, 60, 10, help="How long the benchmark runs.")
    
    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Simulating benchmark run..."):
            try:
                from data_gen.simulator import BenchmarkSimulator
                sim = BenchmarkSimulator(duration_minutes=duration)
                df = sim.generate_base_metrics()
                # Inject some random anomalies for demonstration
                df = sim.inject_anomalies(df, anomaly_ratio=0.1)
                st.session_state["data"] = df
                st.toast(f"Generated {len(df)} records!", icon="✅")
            except Exception as e:
                st.error(f"Simulation failed: {e}")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload Benchmark CSV", type="csv")
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
    st.subheader("📊 Raw Benchmark Metrics")
    st.dataframe(df.head(), use_container_width=True)
    
    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🔍 Analyze with AI", type="primary", use_container_width=True)
    
    if analyze_btn:
        with st.spinner("🤖 AI is analyzing performance..."):
            try:
                # Prepare data for AI
                metrics = df.to_dict(orient="records")
                
                # Explicitly check for required columns here too, to bypass any caching issues
                required_cols = ["cpu_usage", "memory_usage", "disk_io", "network_latency", "throughput"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                     raise ValueError(f"Required benchmark columns missing: {', '.join(missing)}")
                
                results = ai.detect_anomalies(metrics)
                
                # Merge results back to DF
                results_df = pd.DataFrame(results)
                
                # Rename detection column to avoid collision if input already has it
                if "is_anomaly" in results_df.columns:
                    results_df.rename(columns={"is_anomaly": "detected_anomaly"}, inplace=True)
                
                final_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
                
                st.session_state["results"] = final_df
                st.toast("Analysis Complete!", icon="🎉")
            except ValueError as ve:
                st.error(f"❌ **Invalid Data Format:** {ve}")
                st.info("💡 **Tip:** Use the 'Download Sample Template' button in the sidebar to see the required format.")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# --- Results Presentation ---
if "results" in st.session_state:
    res = st.session_state["results"]
    st.divider()
    
    # KPI Metrics
    try:
        avg_score = res["performance_score"].mean()
        # Use detected_anomaly column we created
        anomalies_count = res["detected_anomaly"].sum() 
        total_records = len(res)
        
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("Average Health Score", f"{avg_score:.1f} / 100", delta_color="normal")
        kpi2.metric("Anomalies Detected", f"{anomalies_count}", delta_color="inverse")
        kpi3.metric("Total Data Points", f"{total_records}")
    except KeyError as e:
        st.error(f"Missing expected columns in results: {e}")

    # Charts
    st.subheader("📈 System Performance Trends")
    chart_data = res[["cpu_usage", "memory_usage", "performance_score"]]
    st.line_chart(chart_data)

    # Anomaly Drill-down
    if anomalies_count > 0:
        st.subheader("🚨 Detected Anomalies")
        anomaly_df = res[res["detected_anomaly"] == True]
        
        for idx, row in anomaly_df.iterrows():
            with st.status(f"Anomaly Detected at Index {idx} (Score: {row['performance_score']})", state="error"):
                st.write(f"**Recommendation:** {row['recommendation']}")
                st.json({
                    "CPU": f"{row['cpu_usage']}%",
                    "Memory": f"{row['memory_usage']} MB",
                    "Latency": f"{row['network_latency']} ms",
                    "Throughput": f"{row['throughput']} req/s"
                })
    else:
        st.success("✅ No anomalies detected. System is running smoothly.")

    # Full Data Table
    with st.expander("📂 View Full Analysis Report"):
        st.dataframe(res, use_container_width=True)
