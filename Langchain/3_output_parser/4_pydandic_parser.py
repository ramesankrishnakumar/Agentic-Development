from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")


class Person(BaseModel):
    name: str = Field(description="The name of the person")
    hobbies: list[str] = Field(description="The hobbies of the person")


output_parser = PydanticOutputParser(pydantic_object=Person)

print(output_parser.get_format_instructions())

print("\n" + "-" * 50 + "\n")

template = ChatPromptTemplate([
    HumanMessagePromptTemplate.from_template("{request}\n{format_instructions}"),
])

prompt_values = {
    "request": "Describe a fictional person, include their name and two hobbies.",
    "format_instructions": output_parser.get_format_instructions()
}

print(template.format(**prompt_values))

print("-" * 10 + "parsed response" + "-" * 10)

llm_with_prompt = template | llm | output_parser

response: Person = llm_with_prompt.invoke(input=prompt_values)

print(response)
