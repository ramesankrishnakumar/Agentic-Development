from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

messages = [
    SystemMessage("Keep talking back to me in a random language of your choice"),
    HumanMessage("hi!"),
]

response = model.invoke(messages)

print(response.model_dump())
print(response.content)

messages.append(AIMessage(content=response.content, response_metadata=response.response_metadata))
messages.append(HumanMessage("How are you doing ?"))

print(messages)

response = model.invoke(messages)

print(response.content)

