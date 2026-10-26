import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Configure logging for telemetry and debugging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("MultiLLMWrapper")

class LLMRequest(BaseModel):
    """Standardized request schema for all LLM providers."""
    prompt: str = Field(..., description="The input prompt or user message for the LLM.")
    system_prompt: Optional[str] = Field(default=None, description="Optional system instructions.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int = Field(default=512, ge=1, description="Maximum tokens to generate.")

class LLMResponse(BaseModel):
    """Standardized response schema returned by all LLM wrappers."""
    content: str = Field(..., description="The generated text response from the model.")
    provider: str = Field(..., description="Name of the LLM provider used.")
    model_name: str = Field(..., description="Specific model identifier used.")
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt.")
    completion_tokens: int = Field(..., description="Number of tokens in the generated completion.")
    total_tokens: int = Field(..., description="Total token consumption.")
    latency_seconds: float = Field(..., description="Execution latency in seconds.")
    raw_response: Optional[Dict[str, Any]] = Field(default=None, description="Raw provider payload if needed.")

class BaseLLMWrapper(ABC):
    """Abstract Base Class defining the contract for all LLM wrappers."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generates text response based on the standardized LLMRequest."""
        pass


class OpenAICompatibleWrapper(BaseLLMWrapper):
    """Wrapper supporting OpenAI and Groq (via custom base URL) APIs."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None, provider_name: str = "OpenAI"):
        super().__init__(model_name)
        self.provider_name = provider_name
        
        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY" if provider_name == "OpenAI" else "GROQ_API_KEY")
        if not resolved_api_key:
            raise ValueError(f"CRITICAL: API key for {provider_name} is missing! Please set it in your .env file.")

        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=resolved_api_key,
                base_url=base_url
            )
        except ImportError:
            logger.error("OpenAI package not installed.")
            raise

    def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            latency = time.time() - start_time
            content = response.choices[0].message.content or ""
            usage = response.usage
            
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else (prompt_tokens + completion_tokens)

            return LLMResponse(
                content=content,
                provider=self.provider_name,
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=round(latency, 4)
            )
        except Exception as e:
            logger.error(f"Error generating response from {self.provider_name}: {str(e)}")
            raise e


class AnthropicWrapper(BaseLLMWrapper):
    """Wrapper supporting Anthropic Claude models."""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        super().__init__(model_name)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            logger.error("Anthropic package not installed.")
            raise

    def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        kwargs = {
            "model": self.model_name,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [{"role": "user", "content": request.prompt}]
        }
        if request.system_prompt:
            kwargs["system"] = request.system_prompt

        try:
            response = self.client.messages.create(**kwargs)
            latency = time.time() - start_time
            
            content = response.content[0].text if response.content else ""
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            
            return LLMResponse(
                content=content,
                provider="Anthropic",
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_seconds=round(latency, 4)
            )
        except Exception as e:
            logger.error(f"Error generating response from Anthropic: {str(e)}")
            raise e


class GeminiWrapper(BaseLLMWrapper):
    """Wrapper supporting Google Gemini models via google-genai SDK."""
    
    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        super().__init__(model_name)
        try:
            from google import genai
            self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        except ImportError:
            logger.error("Google GenAI package not installed.")
            raise

    def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        
        config = {
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.system_prompt:
            config["system_instruction"] = request.system_prompt

        try:
            model_id = self.model_name
            if not model_id.startswith("models/"):
                model_id = f"models/{model_id}"

            response = self.client.models.generate_content(
                model=model_id,
                contents=request.prompt,
                config=config
            )
            latency = time.time() - start_time
            
            content = response.text or ""
            usage = getattr(response, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
            
            return LLMResponse(
                content=content,
                provider="Google Gemini",
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_seconds=round(latency, 4)
            )
        except Exception as e:
            logger.error(f"Error generating response from Google Gemini: {str(e)}")
            raise e


class OllamaWrapper(BaseLLMWrapper):
    """Wrapper supporting local Ollama models (Zero Cost & Offline)."""
    
    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url.rstrip("/")
        import requests
        self.session = requests.Session()

    def generate(self, request: LLMRequest) -> LLMResponse:
        start_time = time.time()
        endpoint = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            }
        }
        if request.system_prompt:
            payload["system"] = request.system_prompt

        try:
            response = self.session.post(endpoint, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            content = data.get("response", "")
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            
            return LLMResponse(
                content=content,
                provider="Ollama (Local)",
                model_name=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_seconds=round(latency, 4)
            )
        except Exception as e:
            logger.error(f"Error generating response from Ollama: {str(e)}")
            raise e


class LLMFactory:
    """Factory class to instantiate the correct provider wrapper dynamically."""
    
    @staticmethod
    def get_provider(provider_type: str, model_name: str, **kwargs) -> BaseLLMWrapper:
        provider_lower = provider_type.lower()
        if provider_lower == "openai":
            return OpenAICompatibleWrapper(model_name=model_name, provider_name="OpenAI", **kwargs)
        elif provider_lower == "groq":
            return OpenAICompatibleWrapper(
                model_name=model_name, 
                base_url="https://api.groq.com/openai/v1", 
                provider_name="Groq",
                **kwargs
            )
        elif provider_lower == "anthropic":
            return AnthropicWrapper(model_name=model_name, **kwargs)
        elif provider_lower == "gemini":
            return GeminiWrapper(model_name=model_name, **kwargs)
        elif provider_lower == "ollama":
            return OllamaWrapper(model_name=model_name, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider type: {provider_type}")