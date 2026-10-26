from engine import BatchEvaluationEngine
from telemetry import TelemetryAnalyzer

def run_pipeline_test():
    engine = BatchEvaluationEngine()
    
    input_csv = "test_dataset.csv"
    output_csv = "eval_results.csv"

    # Step 1: Execute Batch Pipeline with LLM Judge & Cost Tracker
    engine.process_batch(input_file=input_csv, output_file=output_csv)

    # Step 2: Generate Telemetry Benchmarking & Quality Metrics Report
    TelemetryAnalyzer.generate_report(results_file=output_csv)

if __name__ == "__main__":
    run_pipeline_test()