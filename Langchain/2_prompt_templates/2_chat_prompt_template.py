from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate

load_dotenv()

# In short:
#
# format() = Template + Variables -> Rendered Prompt Messages (No LLM call)
#
# invoke() = Template + Variables -> Rendered Prompt Messages -> Send to LLM -> Get LLM Response (Includes LLM call)

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

template = ChatPromptTemplate(
    [("system", "You are a helpful AI bot. Your name is {name}."),
     ("human", "Hello, how are you doing?"),
     ("ai", "I'm doing well, thanks!"),
     ("human", "{user_input}"),
     ])
prompt_value = template.format(name="Bob", user_input="What is your name?")

response = llm.invoke(prompt_value)

print(response.content)

template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("You are a AI recipe assistant specializes in {specialty}. Your name is {name}."),
    HumanMessagePromptTemplate.from_template("Hello, how are you doing?"),
    AIMessagePromptTemplate.from_template("I'm doing well, thanks!"),
    HumanMessagePromptTemplate.from_template("{user_input}"),
])

print(template.input_variables)
values = {
    "specialty": "Italian",
    "name": "Bob",
    "user_input": "Give me a gluten-free recipe for Lasagna.",
}
llm_with_prompt = template | llm

response = llm_with_prompt.invoke(input=values)

print(response.content)

template = ChatPromptTemplate([
    {
        "role": "system",
        "content": "You are a helpful AI bot. Your name is {name}.",
    },
    {
        "role": "human",
        "content": "Hello, how are you doing?",
    },
    {
        "role": "ai",
        "content": "I'm doing well, thanks!",
    },
    {
        "role": "human",
        "content": "{user_input}",
    },
])

# print(template.input_variables)

llm_with_prompt = template | llm

response = llm_with_prompt.invoke({ "name": "Bob", "user_input": "What is your name?" })

print(response.content)
