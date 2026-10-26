import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")

print("--- Environment Check ---")
if groq_key:
    # Mask the key for security, showing only first 6 and last 4 characters
    masked_key = f"{groq_key[:6]}...{groq_key[-4:]}" if len(groq_key) > 10 else "INVALID_KEY_LENGTH"
    print(f"SUCCESS: GROQ_API_KEY found in environment! Key: {masked_key}")
else:
    print("ERROR: GROQ_API_KEY is NOT found or .env file is missing/incorrectly named!")