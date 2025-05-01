from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import init_chat_model
from langchain.tools import Tool
from dotenv import load_dotenv
load_dotenv()


from langchain_community.agent_toolkits.load_tools import load_tools


llm = init_chat_model(
    "gemini-2.0-flash-001",
    model_provider="google_genai",
    temperature=0
)

tools = load_tools(["llm-math"], llm=llm)

print(tools)

print("\n" + "-" * 50 + "\n")

print(dir(AgentType))

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

agent.invoke({"input": "What is 2 raised to the .345 power?"})