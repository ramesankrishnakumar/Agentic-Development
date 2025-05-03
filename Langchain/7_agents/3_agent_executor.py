from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

from dotenv import load_dotenv
load_dotenv()


llm = init_chat_model(
    "gemini-2.0-flash-001",
    model_provider="google_genai",
    temperature=0
)


@tool(parse_docstring=True)
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Use this to get weather information.

    Args:
    city: The city name. Can be either 'nyc' or 'sf'.
    """
    if city == "nyc":
        return "It might be cloudy in nyc."
    elif city == "sf":
        return "It's always sunny in sf."
    else:
        return f"It's always sunny in {city}."


@tool(parse_docstring=True)
def search_web(query: str) -> list[dict[str, str]]:
    """A tool to search the web using DuckDuckGo. Useful for when you need to answer questions about current events.

    Args:
    query: The search query.
    """
    results = DDGS().text(query, max_results=4)
    return results

tools = [get_weather, search_web]
prompt = ChatPromptTemplate.from_messages(
    [
        # System prompt. The role, task definition, guidelines and all other instructions should go here.
        (
            "system",
            "You are a helpful assistant. Use the tools provided to help the user.\n.Don't use emogies in your response.",
        ),
        # Placeholder for the messages
        (
            "placeholder",
            "{messages}",
        ),
        # Placeholder for scratchpad that the agent will use to store its thought process
        (
            "placeholder",
            "{agent_scratchpad}",
        ),
    ]
)


agent = create_tool_calling_agent(llm, tools, prompt)



agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

response = agent_executor.invoke(
    {"messages": [{"role": "human", "content": "hi! my name is bob"}]}
)
print(response)

response = agent_executor.invoke(
    {"messages": [("human", "what's the weather today in sf?")]}
)
print(response)

response = agent_executor.invoke(
    {
        "messages": [
            ("human", "What is the age of Elon Musk?"),
        ]
    }
)
print(response)