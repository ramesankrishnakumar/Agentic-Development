from langgraph.checkpoint.memory import MemorySaver

from typing import Literal, Union

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

load_dotenv()

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")
checkpointer = MemorySaver()

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["munich"]:
        return "It's 15 degrees Celsius and cloudy."
    else:
        return "It's 32 degrees Celsius and sunny."


# verify tool is working
result = get_weather.invoke("munich")
print(f"get_weather('munich') -> {result}")

tools = [get_weather]
llm_with_tools = llm.bind_tools(tools)

input_messages = [("human", "What is the weather in munich?")]


def call_model(state: MessagesState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    print("*" * 50)
    print(f"LLM Response: {response}")
    print("*" * 50)
    # why just the latest message? and not the entire conversation?
    # we rely on Langgraph provided default reducer function to do that for us
    # Annotated[list[AnyMessage], add_messages], #add_messages in the reducer function
    return {"messages": [response]}


def router(state: MessagesState) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    print("*" * 50)
    print(f"last_message: {last_message}")
    print("*" * 50)
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("returning tools")
        return "tools"

    print("returning END")
    return END


graph = StateGraph(MessagesState)

graph.add_node("agent", call_model)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "agent")

graph.add_conditional_edges("agent", router)
graph.add_edge("tools", "agent")

runnable = graph.compile(checkpointer=checkpointer)

res = runnable.invoke(
    {"messages": [("human", "How is the weather in munich?")]},
    config={"configurable": {"thread_id": 1}},
)
print("#" * 50)
print(res)
print("#" * 50)

runnable.invoke(
    {"messages": [ ("human", "What would you recommend to do in that city then?")]},
    config={"configurable": {"thread_id": 1}},
)
