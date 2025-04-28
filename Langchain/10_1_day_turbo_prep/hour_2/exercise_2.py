# Exercise 2 Solution

import os

from dotenv import load_dotenv
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
import random
from langchain_core.tools.render import ToolsRenderer

my_list = ["Sunny", "Rainy", "Stormy", "Snowy"]

# Load environment variables
load_dotenv()


# 1. Define a simple Python function that acts as a tool
def get_city_get_location_as_array(city: str) -> [str]:
    """Returns lat: str and long: str of the city name as array"""
    if "london" in city.lower():
        return ['44.968046' , '-94.420307']
    elif "paris" in city.lower():
        return ['4.92057 ' , '-90.44786']

    return ['35.929673', '-75.830290']


def get_weather_by_geo_location(location: [str]) -> str:
    return f"The weather at lat: {location[0]} and long {location[1]} is {random.choice(my_list)}"


# 2. Wrap this function as a Langchain Tool
geo_location = Tool(
    name="get_geo_location",
    func=get_city_get_location_as_array,
    description="Given a city name as string, returns a string array of latitude and longitude => geo location",
)

# 2. Wrap this function as a Langchain Tool
weather_in_geo_location = Tool(
    name="get_weather",
    func=get_weather_by_geo_location,
    description="Given a geo location [lat, long] return descriptive weather at that location",
)

# List of tools available to the agent
tools = [geo_location, weather_in_geo_location]

# 3. Initialize a Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001",
                             temperature=0)  # Lower temp for more predictable agent behavior

# Define the prompt for the ReAct agent
# This prompt outline_template is based on the ReAct framework structure
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the outline_template_values question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the outline_template_values to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original outline_template_values question

Begin!

Question: {outline_template_values}
Thought:{agent_scratchpad}
""")

# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)

# Create an AgentExecutor to run the agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # verbose=True shows the agent's thought process

# 4. Give the agent a query
query = "What is the weather in London?"

# 5. Run the agent
print(f"\nRunning agent with query: '{query}'")
try:
    response = agent_executor.invoke({"outline_template_values": query})
    print("\nAgent Final Response:")
    print(response['output'])
except Exception as e:
    print(f"An error occurred: {e}")
