import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables from .env
load_dotenv()

def list_groq_models():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in .env file.")
        return

    client = Groq(api_key=api_key)
    
    try:
        print("\n🔍 Fetching available models from Groq API...")
        response = client.models.list()
        
        print("\n🔥 Available Groq Models for your API Key:")
        print("-" * 50)
        for model in response.data:
            print(f" ✅ {model.id}")
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ Failed to fetch models: {str(e)}")

if __name__ == "__main__":
    list_groq_models()