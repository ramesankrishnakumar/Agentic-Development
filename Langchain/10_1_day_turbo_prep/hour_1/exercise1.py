# Exercise 1 Solution

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Initialize the Gemini LLM
# You might choose a specific model like "gemini-pro"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", api_key=google_api_key)

# Create a simple message
message = HumanMessage(content="What is the capital of France?")

# Make the call to the LLM
try:
    response = llm.invoke([message])
    print("LLM Response:")
    print(response.content)
except Exception as e:
    print(f"An error occurred: {e}")
    print("Please ensure your GOOGLE_API_KEY is correct and the Gemini API is enabled for your project.")