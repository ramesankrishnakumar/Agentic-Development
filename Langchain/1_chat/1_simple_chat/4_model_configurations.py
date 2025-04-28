from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model(
    model="gemini-2.0-flash-001",
    model_provider="google_genai",
    temperature=1,
    max_tokens=100,
    top_p=0.5
)

messages = [
    SystemMessage("You are a helpful assistant named Bob, who answers questions asked to you"),
    HumanMessage("Hi, Tell me an interesting fact about Pluto"),
]

response = model.invoke(messages)
print(response.content)

response = model.invoke(messages)
print(response.content)


print("*****openAI****")

model = init_chat_model(
    model="gpt-4.1-nano",
    model_provider="openai",
    temperature=0,
    max_tokens=100,
    top_p=0.5,
    frequency_penalty=1,
    presence_penalty=1
)
# temperature: From 0 to 1.
#
# The lower the number is, the more deterministic the response will be.
#
# The higher the number is the more creative the response will be, but moe likely to go off topic if it's too high.
#
# Usage: Lower numbers are good for virtual assistants where we need deterministic responses. Higher numbers are good for roleplay or creative tasks like editing stories
#
# top_p: From 0 to 1.
#
# This is a threshold. Let's use 0.1 (10%) as an example. Only the top 10% of probable tokens are considered.
#
# In the case of 0.9 (90%), we've got a larger set of tokens to work with which makes our responses more diverse, less repetitive, but could also lead to less likely tokens which also means: off-topic responses.


# max_tokens: Limits the number of tokens generated in a response. Aside from specifying int he system message to keep answers short, I could control the length of the answer with this property.
#
# frequency_penalty: From -2.0 to 2.0.
#
# Reduces the chance of repeating tokens
#
# presence_penalty: From -2.0 to 2.0.
#
# Increases the chances of introducing new topics and ideas.

response = model.invoke(messages)
print(response.content)

response = model.invoke(messages)
print(response.content)
