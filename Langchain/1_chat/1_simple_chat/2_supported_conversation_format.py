from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

messages = [
    SystemMessage("You are a helpful assistant named Bob"),
    HumanMessage("hi!"),
    AIMessage("Hi, my name is Bob how can I help you?"),
    HumanMessage("What is the capital of India")
]

response = model.invoke(messages)

print(f"Standard message classes response: {response.content}")

print("*" * 50)

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant named Bob"
    },
    # can use human or user
    {
        "role": "user",
        "content": "hi!"
    },

    # can use "ai" or "assistant"

    {
        "role": "assistant",
        "content": "Hi, my name is Bob how can I help you?"
    },
    {
        "role": "human",
        "content": "What is the capital of India"
    },
    {
        "role": "ai",
        "content": "The capital of India is New Delhi."
    },
    {
        "role": "user",
        "content": "what is the size of India in sq km"
    }

]

response = model.invoke(messages)

print(f"dictionary based chat response: {response.content}")
print("*" * 50)

messages = [("system", "You are a helpful assistant named Bob"),
            ("user", "hi!"),
            ("assistant", "Hi, my name is Bob how can I help you?"),
            ("user", "What is the capital of India")]

response = model.invoke(messages)

print(f"tuple based chat response: {response.content}")
print("*" * 50)