import os
import sys
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from engine import BatchProcessor


# --------------------------------------------------
# 1. Load environment variables from .env
# --------------------------------------------------
load_dotenv()

# Keep input/output files in the same folder as this script
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "test_dataset.csv"
OUTPUT_FILE = BASE_DIR / "batch_evaluation_output.csv"


# --------------------------------------------------
# 2. Logging configuration
# --------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger("BatchEvaluationTest")


# --------------------------------------------------
# 3. LLM configuration
# --------------------------------------------------
PRIMARY_PROVIDER = "groq"
PRIMARY_MODEL = "openai/gpt-oss-120b"

# Keep fallback disabled for the first test.
# Enable it later after Ollama is running correctly.
FALLBACK_PROVIDER = None
FALLBACK_MODEL = None

MAX_RETRIES = 2
INITIAL_BACKOFF = 1.5


# --------------------------------------------------
# 4. Main execution
# --------------------------------------------------
def main():

    print("\n" + "=" * 60)
    print("       BATCH EVALUATION PIPELINE")
    print("=" * 60)

    # Check API key
    if PRIMARY_PROVIDER == "groq":
        if not os.getenv("GROQ_API_KEY"):
            print("\nERROR: GROQ_API_KEY was not found.")
            print("Please add your Groq API key to the .env file.")
            sys.exit(1)

    # Check whether dataset exists
    if not INPUT_FILE.exists():
        print(f"\nERROR: Dataset file not found:\n{INPUT_FILE}")
        print("\nPlease place test_dataset.csv beside test_engine.py.")
        sys.exit(1)

    # Validate dataset columns before running the batch
    try:
        dataset_df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"\nERROR: Could not read the dataset: {e}")
        sys.exit(1)

    required_columns = {"id", "prompt"}

    if not required_columns.issubset(dataset_df.columns):
        missing_columns = required_columns - set(dataset_df.columns)

        print("\nERROR: Your dataset is missing required columns.")
        print(f"Missing columns: {sorted(missing_columns)}")
        print("Required columns: id, prompt")
        sys.exit(1)

    if dataset_df.empty:
        print("\nERROR: The dataset contains no rows.")
        sys.exit(1)

    print(f"\nDataset: {INPUT_FILE.name}")
    print(f"Total rows: {len(dataset_df)}")
    print(f"Primary provider: {PRIMARY_PROVIDER}")
    print(f"Primary model: {PRIMARY_MODEL}")
    print(f"Fallback provider: {FALLBACK_PROVIDER or 'Disabled'}")

    # Initialize BatchProcessor
    try:
        print("\nInitializing Batch Processor...")

        processor = BatchProcessor(
            primary_provider=PRIMARY_PROVIDER,
            primary_model=PRIMARY_MODEL,
            fallback_provider=FALLBACK_PROVIDER,
            fallback_model=FALLBACK_MODEL,
            max_retries=MAX_RETRIES,
            initial_backoff=INITIAL_BACKOFF
        )

    except Exception as e:
        print(f"\nERROR: Could not initialize BatchProcessor.\n{e}")
        sys.exit(1)

    # Process the CSV dataset
    try:
        print("\nStarting batch execution...")
        print("-" * 60)

        results_df = processor.process_csv(
            file_path=str(INPUT_FILE),
            output_csv_path=str(OUTPUT_FILE)
        )

    except Exception as e:
        print(f"\nERROR: Batch execution could not complete.\n{e}")
        sys.exit(1)

    # Handle an empty result safely
    if results_df.empty:
        print("\nNo results were returned by the processor.")
        sys.exit(1)

    # Calculate execution summary
    total_rows = len(results_df)

    success_count = (
        results_df["status"].eq("SUCCESS").sum()
        if "status" in results_df.columns else 0
    )

    failed_count = (
        results_df["status"].eq("FAILED").sum()
        if "status" in results_df.columns else total_rows
    )

    print("\n" + "=" * 60)
    print("              BATCH EXECUTION SUMMARY")
    print("=" * 60)

    print(f"Total rows processed : {total_rows}")
    print(f"Successful rows      : {success_count}")
    print(f"Failed rows          : {failed_count}")

    # Display useful result columns
    display_columns = [
        "row_id",
        "status",
        "provider",
        "model_name",
        "latency_seconds",
        "total_tokens"
    ]

    available_columns = [
        column for column in display_columns
        if column in results_df.columns
    ]

    print("\n--- Row-wise Results ---")

    if available_columns:
        print(
            results_df[available_columns].to_string(index=False)
        )
    else:
        print(results_df.to_string(index=False))

    # Display error details for failed rows
    if failed_count > 0 and "status" in results_df.columns:
        failed_rows = results_df[
            results_df["status"] == "FAILED"
        ]

        if "error_message" in failed_rows.columns:
            print("\n--- Failure Details ---")

            for _, row in failed_rows.iterrows():
                print(f"\nRow ID: {row.get('row_id', 'Unknown')}")
                print(f"Error: {row.get('error_message', 'Unknown')}")

    print("\n" + "-" * 60)
    print(f"Detailed output: {OUTPUT_FILE}")

    if failed_count == 0:
        print("\nSUCCESS: All rows completed successfully.")
    elif success_count > 0:
        print("\nPARTIAL SUCCESS: Some rows failed.")
    else:
        print("\nFAILED: No rows completed successfully.")

    print("=" * 60)


# --------------------------------------------------
# 5. Run the script
# --------------------------------------------------
if __name__ == "__main__":
    main()