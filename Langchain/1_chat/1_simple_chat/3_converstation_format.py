from langchain.chat_models import init_chat_model

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

messages = [
    {
        "role": "system",
        "content": "Keep talking back to me in a random language of your choice"
    },
    {
        "role": "user",
        "content": "Hi !!!"
    }
]

response = model.invoke(messages)

print(response.model_dump())

messages.append(
    {
        "role": "ai",
        "content": response.content,
        "response_metadata": response.response_metadata
    }
)

messages.append(
    {
        "role": "user",
        "content": "How are you feeling today ?"
    }
)

print(messages)

response = model.invoke(messages)

print(response.model_dump())
