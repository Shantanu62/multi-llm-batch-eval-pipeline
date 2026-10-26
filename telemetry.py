import pandas as pd
import numpy as np
from typing import Dict, Any

class TelemetryAnalyzer:
    """Enterprise Latency Benchmarking, Cost Analytics, and Quality Metric Summarizer."""

    @staticmethod
    def generate_report(results_file: str) -> Dict[str, Any]:
        df = pd.read_csv(results_file)
        
        total_requests = len(df)
        successful_requests = len(df[df['status'] == 'SUCCESS'])
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0.0

        latencies = df[df['status'] == 'SUCCESS']['latency'].tolist()
        
        if latencies:
            p50 = round(float(np.percentile(latencies, 50)), 4)
            p95 = round(float(np.percentile(latencies, 95)), 4)
            p99 = round(float(np.percentile(latencies, 99)), 4)
            avg_latency = round(float(np.mean(latencies)), 4)
        else:
            p50 = p95 = p99 = avg_latency = 0.0

        # Token & Cost Metrics
        total_tokens = int(df['total_tokens'].sum())
        avg_tokens = round(float(df['total_tokens'].mean()), 2) if total_requests > 0 else 0.0
        total_cost = round(float(df['cost_usd'].sum()), 6)

        # Quality & Judge Metrics
        avg_correctness = round(float(df[df['status'] == 'SUCCESS']['correctness_score'].mean()), 2) if successful_requests > 0 else 0.0
        safety_pass_rate = round((len(df[df['safety_pass'] == True]) / total_requests) * 100, 2) if total_requests > 0 else 0.0

        report = {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_p99": p99,
            "avg_latency": avg_latency,
            "total_tokens": total_tokens,
            "avg_tokens": avg_tokens,
            "total_cost_usd": total_cost,
            "avg_correctness_score": avg_correctness,
            "safety_pass_rate": safety_pass_rate
        }

        print("\n========================================================")
        print("📊 ENTERPRISE LLM BENCHMARK & TELEMETRY REPORT 📊")
        print("========================================================")
        print(f" Total Requests Evaluated : {total_requests}")
        print(f" Success Rate             : {success_rate:.2f}%")
        print(f"--------------------------------------------------------")
        print(f" ⏱️ Latency P50 (Median)   : {p50} seconds")
        print(f" ⏱️ Latency P95            : {p95} seconds")
        print(f" ⏱️ Latency P99            : {p99} seconds")
        print(f" ⏱️ Avg Latency            : {avg_latency} seconds")
        print(f"--------------------------------------------------------")
        print(f" 🔠 Total Tokens Used      : {total_tokens}")
        print(f" 🔠 Avg Tokens / Request   : {avg_tokens}")
        print(f" 💵 Total Estimated Cost   : ${total_cost:.6f} USD")
        print(f"--------------------------------------------------------")
        print(f" ⭐ Avg Correctness Score  : {avg_correctness} / 5.0")
        print(f" 🛡️ Safety Pass Rate       : {safety_pass_rate:.2f}%")
        print("========================================================\n")

        return report