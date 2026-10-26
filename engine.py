import os
import time
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from wrapper import LLMFactory, LLMRequest, LLMResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("BatchProcessor")

class BatchItemResult(BaseModel):
    row_id: int = Field(..., description="Unique identifier from the dataset row.")
    prompt: str = Field(..., description="The input prompt executed.")
    provider: str = Field(..., description="LLM provider used.")
    model_name: str = Field(..., description="Specific model used.")
    content: Optional[str] = Field(default=None, description="Generated response content.")
    prompt_tokens: int = Field(default=0, description="Input token count.")
    completion_tokens: int = Field(default=0, description="Output token count.")
    total_tokens: int = Field(default=0, description="Total token consumption.")
    latency_seconds: float = Field(default=0.0, description="Execution latency in seconds.")
    status: str = Field(..., description="Execution status: SUCCESS or FAILED.")
    error_message: Optional[str] = Field(default=None, description="Error details if execution failed.")

class BatchProcessor:
    """Engine supporting primary and fallback LLM routing with retry logic."""
    
    def __init__(
        self, 
        primary_provider: str, 
        primary_model: str, 
        fallback_provider: Optional[str] = None, 
        fallback_model: Optional[str] = None,
        max_retries: int = 3, 
        initial_backoff: float = 2.0
    ):
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        
        # Initialize Primary Provider Wrapper
        self.primary_wrapper = LLMFactory.get_provider(provider_type=primary_provider, model_name=primary_model)
        logger.info(f"Primary Provider Initialized: {primary_provider} ({primary_model})")
        
        # Initialize Fallback Provider Wrapper (Optional)
        self.fallback_wrapper = None
        if fallback_provider and fallback_model:
            try:
                self.fallback_wrapper = LLMFactory.get_provider(provider_type=fallback_provider, model_name=fallback_model)
                logger.info(f"Fallback Provider Initialized: {fallback_provider} ({fallback_model})")
            except Exception as e:
                logger.warning(f"Could not initialize fallback provider: {e}")

    def _execute_with_fallback(self, request: LLMRequest) -> LLMResponse:
        """Tries primary model with retries; if it fails, falls back to secondary model."""
        backoff = self.initial_backoff
        last_exception = None

        # 1. Try Primary Model First
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.primary_wrapper.generate(request)
            except Exception as e:
                last_exception = e
                logger.warning(f"Primary Model attempt {attempt}/{self.max_retries} failed: {str(e)}")
                if attempt < self.max_retries:
                    time.sleep(backoff * (2 ** (attempt - 1)))

        # 2. If Primary Fails Completely, Switch to Fallback Model
        if self.fallback_wrapper:
            logger.warning("Primary model exhausted all retries. Switching to Fallback Provider...")
            for attempt in range(1, self.max_retries + 1):
                try:
                    return self.fallback_wrapper.generate(request)
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Fallback Model attempt {attempt}/{self.max_retries} failed: {str(e)}")
                    if attempt < self.max_retries:
                        time.sleep(backoff * (2 ** (attempt - 1)))

        # If both fail, raise the final exception
        raise last_exception

    def process_csv(self, file_path: str, output_csv_path: Optional[str] = None) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at: {file_path}")

        df = pd.read_csv(file_path)
        required_columns = {"id", "prompt"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Dataset must contain 'id' and 'prompt' columns.")

        results: List[Dict[str, Any]] = []

        for index, row in df.iterrows():
            row_id = int(row["id"])
            prompt = str(row["prompt"])
            system_prompt = str(row.get("system_prompt", "You are a helpful AI assistant."))
            
            logger.info(f"Processing Row ID {row_id}...")

            llm_request = LLMRequest(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=256
            )

            try:
                # Execute with automatic fallback capability
                response = self._execute_with_fallback(llm_request)
                
                result_item = BatchItemResult(
                    row_id=row_id,
                    prompt=prompt,
                    provider=response.provider,
                    model_name=response.model_name,
                    content=response.content,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    latency_seconds=response.latency_seconds,
                    status="SUCCESS"
                )
            except Exception as e:
                result_item = BatchItemResult(
                    row_id=row_id,
                    prompt=prompt,
                    provider="Multiple (Failed)",
                    model_name="N/A",
                    status="FAILED",
                    error_message=str(e)
                )

            results.append(result_item.model_dump())
            time.sleep(0.5)

        results_df = pd.DataFrame(results)
        if output_csv_path:
            results_df.to_csv(output_csv_path, index=False)
            logger.info(f"Results exported to {output_csv_path}")

        return results_df