from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.tools import tool

load_dotenv(find_dotenv())

llm = init_chat_model("gemini-1.5-flash-8b", model_provider="google_genai")
# llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

print("*" * 10 + "no tool calling" + "*" * 10)
print(llm.invoke("Hi what is the weather in San Francisco, today ?").content)
print(llm.invoke("Hi what's the time now in California?").content)


@tool
def get_time_for_timezone(timezone: str = "US/Pacific") -> str:
    """Returns the current system time for the given timezone. e.g. US/Pacific """
    from datetime import datetime
    # calculate the time in timezone
    import pytz
    tz = pytz.timezone(timezone)
    return datetime.now().astimezone(tz).strftime("%H:%M:%S")

@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["munich"]:
        return "It's 15 degrees Celsius and cloudy."
    else:
        return "It's 32 degrees Celsius and sunny."


@tool
def check_seating_availability(location: str, seating_type: str):
    """Call to check seating availability."""
    if location.lower() == "munich" and seating_type.lower() == "outdoor":
        return "Yes, we still have seats available outdoors."
    elif location.lower() == "munich" and seating_type.lower() == "indoor":
        return "Yes, we have indoor seating available."
    else:
        return "Sorry, seating information for this location is unavailable."


tools = [get_time_for_timezone, get_weather, check_seating_availability]


llm_with_tools = llm.bind_tools(tools)

print("*" * 10 + "tool calling" + "*" * 10)

tool_mapping = {
    "get_weather": get_weather,
    "check_seating_availability": check_seating_availability,
    "get_time_for_timezone": get_time_for_timezone,
}

messages = [
    HumanMessage(
        "How will the weather be in munich today? Do you still have seats outdoor available?"
    )
]
response = llm_with_tools.invoke(messages)

messages.append(response)





if hasattr(response, "tool_calls"):
    print(response.tool_calls)
    print("*" * 50)
    for tool_call in response.tool_calls:
        tool = tool_mapping[tool_call["name"].lower()]
        tool_output = tool.invoke(tool_call["args"])
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    print(llm_with_tools.invoke(messages).content)
else :
    print(response.content)
