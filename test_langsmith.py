from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import traceback

# Load environment variables from .env file
load_dotenv()

print("Attempting to initialize ChatOpenAI...")
try:
    llm = ChatOpenAI()  # This will use your OPENAI_API_KEY from .env
    print("ChatOpenAI initialized successfully.")

    print("Invoking LLM...")
    response = llm.invoke("Hello, world! This is a test from LangSmith setup.")
    print("LLM Invocation successful.")
    print(f"Response content: {response.content[:300]}...")
    print("---")
    print("SUCCESS: If you see this, the script ran and got a response.")
    print("Please check your LangSmith project 'toefl-project' for the trace.")

except Exception as e:
    print(f"ERROR: An error occurred during the LangSmith test: {e}")
    traceback.print_exc()