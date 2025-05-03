from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate


# Load environment variables
load_dotenv()


# 1. Define a simple Python function that acts as a tool
def get_weather(city: str) -> str:
    """Returns a hardcoded weather description for a given city."""
    if "london" in city.lower():
        return "The weather in London is cloudy with a chance of rain."
    elif "paris" in city.lower():
        return "Paris is sunny and warm."
    else:
        return f"Weather information not available for {city}."

# 2. Wrap this function as a Langchain Tool
weather_tool = Tool(
    name="get_weather",
    func=get_weather,
    description="Useful for getting the weather information for a city. Input should be a city name.",
)

# List of tools available to the agent
tools = [weather_tool]

# 3. Initialize a Gemini LLM
llm = init_chat_model(
    "gemini-2.0-flash-001",
    model_provider="google_genai",
    temperature=0
)

# Define the prompt for the ReAct agent
# This prompt template is based on the ReAct framework structure
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")


# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True) # verbose=True shows the agent's thought process

# 4. Give the agent a query
query = "What is the weather in London?"

# 5. Run the agent
print(f"\nRunning agent with query: '{query}'")
try:
    response = agent_executor.invoke({"input": query})
    print("\nAgent Final Response:")
    print(response['output'])
except Exception as e:
    print(f"An error occurred: {e}")
