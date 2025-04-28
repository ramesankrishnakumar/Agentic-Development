from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, SystemMessage

from dotenv import load_dotenv
load_dotenv()

chat_message_history = SQLChatMessageHistory(
    session_id="test_session_id", connection="sqlite:///langchain.db"
)

# session_id is the primary key
chat_message_history.add_message(SystemMessage("You are a helpful assistant named KK, what answers question with enthusiasm"))
chat_message_history.add_user_message("Hello")
chat_message_history.add_ai_message("Hi, my name is KK how can I help you?")
chat_message_history.add_user_message("What is the capital of India and how many times did India win cricket world cup?")


print(chat_message_history.messages)
print("\n")

from langchain.chat_models import init_chat_model
model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")


response = model.invoke(chat_message_history.messages)

print(response.content)
print("\n")

chat_message_history.add_ai_message(response.content)

print("Storing 1_chat response in persistence")
print("\n")
print(chat_message_history.messages)



chat_message_history.clear()