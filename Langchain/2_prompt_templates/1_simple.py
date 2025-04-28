from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

# llm = GoogleGenerativeAI(model="gemini-2.0-flash-001")
llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

template = """
    Write a {tone} email to {company} 
    expressing interest in the {position} position, 
    mentioning {skill} as a key strength. 
    Keep it to {lines} lines max
"""


llm_with_prompt = PromptTemplate.from_template(template) | llm

response = llm_with_prompt.invoke(input = {
    "tone" : "professional & confidant",
    "company": "Intuit.com",
    "position" : "Senior Software Engineer",
    "skill" : "Autonomous Agent Development",
    "lines" : 10
})

print(response.content)


