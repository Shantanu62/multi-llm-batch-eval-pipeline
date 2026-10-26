import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from engine import BatchEvaluationEngine

# Page Configuration
st.set_page_config(
    page_title="NexusLLM | Enterprise LLMOps & Telemetry Intelligence",
    page_icon="⚡",
    layout="wide"
)

# World-Class Custom CSS Styling (Glassmorphism & Cyber-Dark Theme)
st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0f172a 0%, #090d16 100%);
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(51, 65, 85, 0.8);
        padding: 20px;
        border-radius: 14px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #6366f1;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.5);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(51, 65, 85, 0.5);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94a3b8;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("### ⚡ NEXUS-LLM // ENTERPRISE TELEMETRY & OBSERVABILITY SUITE")
st.markdown("Real-time LLM benchmarking, statistical percentile latency auditing, cost optimization matrices, and automated LLM-as-a-Judge governance.")

# Sidebar Controls
st.sidebar.markdown("## 🎛️ Pipeline Engine")
input_csv = st.sidebar.text_input("Dataset Path", value="test_dataset.csv")
output_csv = st.sidebar.text_input("Telemetry Output", value="eval_results.csv")

if st.sidebar.button("🚀 Execute Batch Pipeline", type="primary"):
    with st.spinner("Running high-performance concurrent evaluation..."):
        engine = BatchEvaluationEngine()
        engine.process_batch(input_file=input_csv, output_file=output_csv)
    st.sidebar.success("Execution completed successfully!")

if os.path.exists(output_csv):
    df = pd.read_csv(output_csv)
    
    # Scientific Filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔬 Statistical Controls")
    min_correctness = st.sidebar.slider("Min Correctness Threshold", 1, 5, 1)
    max_latency_cap = st.sidebar.slider("Max Latency Tolerance (s)", 1.0, 15.0, 10.0)
    
    filtered_df = df[(df['correctness_score'] >= min_correctness) & (df['latency'] <= max_latency_cap)]

    # Tabs for World-Class Navigation
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Telemetry", 
        "📈 Deep-Dive Analytics & Outliers", 
        "⚖️ LLM-as-a-Judge Governance", 
        "📋 Raw Payload Explorer"
    ])

    # --- TAB 1: EXECUTIVE TELEMETRY ---
    with tab1:
        st.markdown("#### High-Level Performance Indicators")
        
        total_requests = len(df)
        success_rate = (len(df[df['status'] == 'SUCCESS']) / total_requests) * 100 if total_requests > 0 else 0
        latencies = df[df['status'] == 'SUCCESS']['latency'].tolist()
        
        p50 = round(float(pd.Series(latencies).quantile(0.5)), 4) if latencies else 0.0
        p95 = round(float(pd.Series(latencies).quantile(0.95)), 4) if latencies else 0.0
        p99 = round(float(pd.Series(latencies).quantile(0.99)), 4) if latencies else 0.0
        avg_latency = round(float(pd.Series(latencies).mean()), 4) if latencies else 0.0
        
        total_cost = round(float(df['cost_usd'].sum()), 6) if 'cost_usd' in df.columns else 0.0
        avg_score = round(float(df['correctness_score'].mean()), 2) if total_requests > 0 else 0.0
        safety_rate = round((len(df[df['safety_pass'] == True]) / total_requests) * 100, 2) if total_requests > 0 else 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Evaluations", total_requests, delta="Batch Live")
        c2.metric("Success Rate", f"{success_rate:.1f}%", delta="Optimal")
        c3.metric("Mean Latency", f"{avg_latency}s")
        c4.metric("Cumulative Cost", f"${total_cost:.6f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("⏱️ P50 (Median)", f"{p50}s")
        c6.metric("⏱️ P95 Percentile", f"{p95}s")
        c7.metric("⏱️ P99 Percentile", f"{p99}s", delta="SLA Safe" if p99 < 8.0 else "High Tail")

        c8, c9 = st.columns(2)
        c8.metric("⭐ Mean Correctness", f"{avg_score} / 5.0")
        c9.metric("🛡️ Safety Compliance", f"{safety_rate}%")

    # --- TAB 2: DEEP-DIVE ANALYTICS ---
    with tab2:
        st.markdown("#### Statistical Latency & Token Economics")
        
        col_a, col_b = st.columns(2)
        with col_a:
            fig_box = px.box(
                df, y='latency', 
                title="Latency Distribution & Outlier Spread",
                color_discrete_sequence=['#6366f1']
            )
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_box, use_container_width=True)
            
        with col_b:
            if 'total_tokens' in df.columns and 'cost_usd' in df.columns:
                fig_scatter = px.scatter(
                    df, x='total_tokens', y='cost_usd', 
                    color='correctness_score',
                    title="Token Throughput vs Cost Efficiency Matrix",
                    color_continuous_scale='Viridis'
                )
                fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
                st.plotly_chart(fig_scatter, use_container_width=True)

    # --- TAB 3: LLM-AS-A-JUDGE GOVERNANCE ---
    with tab3:
        st.markdown("#### Automated Quality Audit & Tone Distribution")
        
        if 'tone_analysis' in df.columns:
            tone_df = df['tone_analysis'].value_counts().reset_index()
            tone_df.columns = ['Tone', 'Frequency']
            fig_pie = px.pie(
                tone_df, names='Tone', values='Frequency', 
                hole=0.4, title="Model Response Tone Breakdown",
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc')
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("#### 📝 Detailed Audit Trail")
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Audit ID: #{idx+1} | Score: {row.get('correctness_score')}/5 | Safety: {row.get('safety_pass')}"):
                st.write(f"**Prompt:** {row.get('prompt')}")
                st.write(f"**Generated Response:** {row.get('response')}")
                st.write(f"**Judge Reasoning:** {row.get('reasoning')}")

    # --- TAB 4: RAW PAYLOAD EXPLORER ---
    with tab4:
        st.markdown("#### Telemetry Data Grid")
        st.dataframe(filtered_df, use_container_width=True)
        
        csv_export = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Certified Telemetry Report",
            data=csv_export,
            file_name="certified_telemetry_audit.csv",
            mime="text/csv"
        )
else:
    st.info("ℹ️ Sidebar se 'Execute Batch Pipeline' par click karein taaki live telemetry data generate ho sake.")