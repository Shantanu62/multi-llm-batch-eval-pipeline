from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Available Models supporting generateContent:")
for model in client.models.list():
    if "generateContent" in getattr(model, "supported_generation_methods", []):
        print(f"- {model.name}")