from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

import time

from dotenv import load_dotenv
load_dotenv()



model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

messages = [
    SystemMessage("Keep talking back to me in a random language of your choice "),
    HumanMessage("hi!"),
]
set_llm_cache(InMemoryCache())



start_time = time.perf_counter()

response = model.invoke(messages)
print(response.content)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")

print("Let's see if the new response is from cache")


start_time = time.perf_counter()
response = model.invoke(messages)
print(response.content)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")


print("Let me change the 2_prompt_templates")

messages = [
    SystemMessage("Keep talking back to me in a random language of your choice "),
    HumanMessage("I'm KK, What is your name ?"),
]

start_time = time.perf_counter()
response = model.invoke(messages)
print(response.content)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")





