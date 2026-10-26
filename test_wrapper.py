import os
from dotenv import load_dotenv
from wrapper import LLMFactory, LLMRequest

# Load environment variables from the .env file securely
load_dotenv()

if __name__ == "__main__":
    try:
        print("Initializing Groq client using secure environment variables...")
        
        # The wrapper automatically reads GROQ_API_KEY from environment variables
        wrapper = LLMFactory.get_provider(
            provider_type="groq", 
            model_name="llama-3.1-8b-instant"
        )
        
        request = LLMRequest(
            prompt="Explain Agentic AI in 2 concise sentences.",
            system_prompt="You are a senior AI engineer.",
            temperature=0.3,
            max_tokens=100
        )
        
        response = wrapper.generate(request)
        print("\n--- Response Received Successfully ---")
        print(f"Provider: {response.provider} ({response.model_name})")
        print(f"Content: {response.content}")
        print(f"Tokens Used: {response.total_tokens} (Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens})")
        print(f"Latency: {response.latency_seconds}s")
        
    except Exception as e:
        print(f"Test failed (check your .env configuration and API key): {e}")