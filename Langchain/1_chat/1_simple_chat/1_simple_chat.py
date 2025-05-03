from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# gpt_model = init_chat_model("gpt-4.1-nano", model_provider="openai")
#
# messages = [
#     SystemMessage("Translate the following from English into Italian"),
#     HumanMessage("hi!"),
# ]

# gpt_response = gpt_model.invoke(messages)
# print(gpt_response.model_dump())
# print(gpt_response.content)


# refer https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html
gemini_model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

messages = [
    SystemMessage("Keep talking back to me in a random language of your choice"),
    HumanMessage("hi!"),
]

gemini_response = gemini_model.invoke(messages)

# prints the response format
print(gemini_response.model_dump())
print(gemini_response.content)

messages.append(AIMessage(content=gemini_response.content, response_metadata=gemini_response.response_metadata))
messages.append(HumanMessage("How are you doing ?"))

print(messages)

response = gemini_model.invoke(messages)

print(response.content)
