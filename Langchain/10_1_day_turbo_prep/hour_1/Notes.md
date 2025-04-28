# Hour 1: Introduction to LLMs, Langchain Basics, and Gemini Setup

## Concepts:

- **Large Language Models (LLMs)**: Models trained on vast amounts of text data, capable of understanding and generating human-like text, translating languages, writing different kinds of creative content, and answering your questions in an informative way.
- **Langchain**: A framework designed to simplify the development of applications using large language models. It provides abstractions for working with LLMs, chaining calls together, and creating agents that can interact with their environment.

### Key Langchain Components:

- **LLMs**: The language models themselves (e.g., Gemini).
- **Prompts**: Templates for guiding the LLM's output.
- **Chains**: Sequences of calls to LLMs or other utilities.
- **Agents**: Components that use an LLM to decide which actions to take based on user input and observations, often interacting with Tools.
- **Tools**: Functions that agents can call to interact with the outside world (e.g., searching the web, running code, accessing databases).
- **Gemini API**: Accessing Google's Gemini models programmatically.

### Notes:

- Langchain provides a standard interface for many LLMs, making it easier to swap models.
- Agents are powerful because they introduce a loop: Observe -> Decide -> Act -> Observe...
- You'll need a Google Cloud Project and enable the Gemini API to use it. Be mindful of API costs.

## Exercise 1: Basic LLM Call with Langchain and Gemini

1. Install the necessary libraries:
   ```bash
   pip install langchain-google-genai python-dotenv
   ```
2. Get your Google Cloud Project ID and API Key.
3. Create a .env file in your project directory and add `GOOGLE_API_KEY='YOUR_API_KEY'`.
4. Write a Python script that:
   - Loads the API key from the .env file.
   - Initializes a Gemini LLM model using Langchain.
   - Makes a simple call to the LLM (e.g., ask \\\"What is the capital of France?\\\").
   - Prints the response.

### Quiz 1: Langchain Fundamentals

1. What is the primary purpose of the Langchain framework?
2. Name three key components of Langchain.
3. What is the role of an Agent in Langchain?
4. Why would you use a .env file for your API key?

## Capstone Integration (Hour 1):

- **Define the core problem the Research Assistant solves (answering questions based on research).**
- Set up the basic project structure and environment with the necessary libraries and API key setup.
