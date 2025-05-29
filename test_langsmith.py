import os
import google.generativeai as genai
from dotenv import load_dotenv
import traceback

# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

print("Attempting to initialize Gemini model...")
try:
    # Initialize the Gemini model
    model = genai.GenerativeModel('gemini-pro')
    print("Gemini model initialized successfully.")

    print("Invoking model...")
    response = model.generate_content("Hello, world! This is a test from LangSmith setup.")
    print("Model invocation successful.")
    print(f"Response content: {response.text[:300]}...")
    print("---")
    print("SUCCESS: If you see this, the script ran and got a response.")
    print("Please check your LangSmith project 'toefl-project' for the trace.")

except Exception as e:
    print(f"ERROR: An error occurred during the test: {e}")
    traceback.print_exc()