# 🚀 Python API & Multi-LLM Automated Batch Evaluation Pipeline

A production-grade, enterprise-ready Python backend system designed to route prompts, execute asynchronous batch evaluations, benchmark latency metrics, track real-time token costs, and automate LLM-as-a-Judge scoring across multiple LLM providers.

---

## 🛠️ Tech Stack & Architecture
* **Language:** Python 3.10+
* **Data Handling:** Pandas, Pydantic (v2)
* **LLM Providers Supported:** 
  * Groq (Blazing fast, OpenAI-compatible)[cite: 1]
  * OpenAI GPT-4o / GPT-3.5[cite: 1]
  * Anthropic Claude (v3.5 Sonnet)[cite: 1]
  * Google Gemini (Gemini Flash / Pro)[cite: 1]
  * Ollama (Local offline models)[cite: 1]
* **Architecture Design:** Abstract Factory Pattern for polymorphic LLM routing.

---

## 📂 Project Structure
```text
project-2-batch-eval-pipeline/
│
├── wrapper.py           # Unified Multi-LLM Abstract Factory Wrapper
├── requirements.txt     # Project Dependencies
├── .env                 # Local Environment Variables (Ignored in Git)
├── .env.example         # Template for Environment Variables
├── .gitignore           # Git Ignore Rules
└── test_wrapper.py      # Verification and Test Script