from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

template = """
    Write a {tone} email to {company} 
    expressing interest in the {position} position, 
    mentioning {skill} as a key strength. 
    Keep it to {lines} lines max
"""

prompt_template = PromptTemplate.from_template(template)

llm_with_prompt = prompt_template | llm

response = llm_with_prompt.invoke(input={
    "tone": "professional & confidant",
    "company": "Intuit.com",
    "position": "Senior Software Engineer",
    "skill": "Autonomous Agent Development",
    "lines": 10
})

print(response.content)

print("\n" + "-" * 10 + "save the prompt and load this" + "-" * 10 + "\n")

# lets save the prompt and load this

prompt_template.save("prompt_template.json")

from langchain_core.prompts import load_prompt

loaded_prompt = load_prompt("prompt_template.json")

llm_with_prompt = loaded_prompt | llm

response = llm_with_prompt.invoke(input={
    "tone": "professional & confidant",
    "company": "AWS",
    "position": "Senior Software Engineer",
    "skill": "Autonomous Agent Development",
    "lines": 10
})

print(response.content)


