import os
import pandas as pd
import numpy as np

def analyze_batch_results(output_csv_path: str = "batch_evaluation_output.csv"):
    """Reads batch evaluation results and computes production telemetry metrics."""
    if not os.path.exists(output_csv_path):
        print(f"Error: Output file not found at {output_csv_path}.")
        return

    df = pd.read_csv(output_csv_path)
    
    print("=" * 60)
    print("       MULTI-LLM BATCH EVALUATION TELEMETRY REPORT       ")
    print("=" * 60)
    
    total_requests = len(df)
    if total_requests == 0:
        print("Dataset is empty.")
        return

    success_df = df[df["status"] == "SUCCESS"]
    failed_count = len(df[df["status"] == "FAILED"])
    success_rate = (len(success_df) / total_requests) * 100

    print(f"Total Requests Processed : {total_requests}")
    print(f"Successful Executions    : {len(success_df)}")
    print(f"Failed Executions        : {failed_count}")
    print(f"Pipeline Success Rate    : {success_rate:.2f}%")
    print("-" * 60)

    if not success_df.empty:
        latencies = success_df["latency_seconds"].values
        tokens = success_df["total_tokens"].values

        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        avg_latency = np.mean(latencies)

        print("--- Latency Benchmarks (Seconds) ---")
        print(f"  Mean Latency       : {avg_latency:.4f}s")
        print(f"  P50 (Median)       : {p50_latency:.4f}s")
        print(f"  P95 Percentile     : {p95_latency:.4f}s")
        print(f"  P99 Percentile     : {p99_latency:.4f}s")
        print("-" * 60)

        total_tokens_used = np.sum(tokens)
        avg_tokens = np.mean(tokens)
        print("--- Token Consumption & Throughput ---")
        print(f"  Total Tokens Used  : {int(total_tokens_used)}")
        print(f"  Avg Tokens / Prompt: {avg_tokens:.1f}")
    else:
        print("No successful executions found to calculate metrics.")

    print("=" * 60)

if __name__ == "__main__":
    analyze_batch_results()