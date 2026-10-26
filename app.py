import streamlit as st
import pandas as pd
import os
from engine import BatchEvaluationEngine
from telemetry import TelemetryAnalyzer

# Page Configuration
st.set_page_config(
    page_title="Multi-LLM Batch Eval & Telemetry Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("🚀 Enterprise Multi-LLM Batch Evaluation & Telemetry Dashboard")
st.markdown("Benchmark LLM outputs, track latency percentiles (P50/P95/P99), monitor token costs, and evaluate quality using automated **LLM-as-a-Judge** scoring.")

# Sidebar Configuration Controls
st.sidebar.header("⚙️ Pipeline Configuration")
input_csv = st.sidebar.text_input("Input Dataset Path", value="test_dataset.csv")
output_csv = st.sidebar.text_input("Output Results Path", value="eval_results.csv")

if st.sidebar.button("🚀 Run Batch Evaluation", type="primary"):
    with st.spinner("Executing batch evaluation pipeline... Please wait."):
        engine = BatchEvaluationEngine()
        engine.process_batch(input_file=input_csv, output_file=output_csv)
    st.success("✅ Evaluation completed successfully!")

# Display Results if Output File Exists
if os.path.exists(output_csv):
    st.markdown("---")
    st.header("📊 Evaluation Telemetry & Analytics Report")
    
    df = pd.read_csv(output_csv)
    
    # Calculate Metrics for UI Cards
    total_requests = len(df)
    successful_requests = len(df[df['status'] == 'SUCCESS'])
    success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0.0
    
    latencies = df[df['status'] == 'SUCCESS']['latency'].tolist()
    p50 = round(float(pd.Series(latencies).quantile(0.5)), 4) if latencies else 0.0
    p95 = round(float(pd.Series(latencies).quantile(0.95)), 4) if latencies else 0.0
    p99 = round(float(pd.Series(latencies).quantile(0.99)), 4) if latencies else 0.0
    avg_latency = round(float(pd.Series(latencies).mean()), 4) if latencies else 0.0
    
    total_tokens = int(df['total_tokens'].sum())
    total_cost = round(float(df['cost_usd'].sum()), 6)
    avg_correctness = round(float(df[df['status'] == 'SUCCESS']['correctness_score'].mean()), 2) if successful_requests > 0 else 0.0
    safety_pass_rate = round((len(df[df['safety_pass'] == True]) / total_requests) * 100, 2) if total_requests > 0 else 0.0

    # Top Metric Cards Row 1
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", total_requests)
    col2.metric("Success Rate", f"{success_rate:.1f}%")
    col3.metric("Avg Latency", f"{avg_latency}s")
    col4.metric("Total Estimated Cost", f"${total_cost:.6f}")

    # Metric Cards Row 2 (Latencies)
    col5, col6, col7 = st.columns(3)
    col5.metric("⏱️ P50 Latency", f"{p50}s")
    col6.metric("⏱️ P95 Latency", f"{p95}s")
    col7.metric("⏱️ P99 Latency", f"{p99}s")

    # Metric Cards Row 3 (Quality Metrics)
    col8, col9 = st.columns(2)
    col8.metric("⭐ Avg Correctness Score", f"{avg_correctness} / 5.0")
    col9.metric("🛡️ Safety Pass Rate", f"{safety_pass_rate}%")

    st.markdown("---")
    st.subheader("📋 Detailed Evaluation Results Table")
    st.dataframe(df, use_container_width=True)
else:
    st.info("ℹ️ Click the **'Run Batch Evaluation'** button in the sidebar to execute the pipeline and render telemetry metrics.")