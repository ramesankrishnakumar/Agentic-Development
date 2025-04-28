from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

# Pass a sequence of prompts to the model and return model generations.
# This method should make use of batched calls for models that expose a batched API.
# Use this method when you want to:
#   take advantage of batched calls,
#   need more output from the model than just the top generated value,
#   are building chains that are agnostic to the underlying language model
#   type (e.g., pure text completion models vs 1_chat models).

response = model.generate([
    [
        SystemMessage("Keep talking back to me in a random language of your choice"),
        HumanMessage("hi!")
    ],

    [
        SystemMessage("Keep talking back to me like a confused toddler"),
        HumanMessage("hi!")
    ]

])

print(response.model_dump())
print(response.llm_output)
print(response.generations[0][0].text)
