# 🚀 Enterprise Multi-LLM Batch Evaluation & Telemetry Pipeline

An enterprise-ready benchmarking and evaluation tool designed to process batch prompts across Large Language Models, track production-grade telemetry, calculate precise cost analytics, and automate qualitative assessment using an **LLM-as-a-Judge** framework.

---

## 🏗️ Architectural Flow

```mermaid
graph TD
    A[Input Dataset CSV] -->|Batch Engine| B(LLM Provider Wrapper / Groq API)
    B -->|Model Response & Tokens| C(Telemetry & Cost Analyzer)
    B -->|Model Output & Prompt| D(LLM-as-a-Judge Evaluator)
    C --> E[Structured Telemetry Report]
    D --> E
    E --> F[Export to eval_results.csv]
    E --> G[Interactive Streamlit Dashboard]