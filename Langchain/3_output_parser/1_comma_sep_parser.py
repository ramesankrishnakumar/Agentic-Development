from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()
llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")
output_parser = CommaSeparatedListOutputParser()

print(output_parser.get_format_instructions())

reply = "apple,banana,orange"
print(output_parser.parse(reply))

template = ChatPromptTemplate([
    HumanMessagePromptTemplate.from_template("{request}\n{format_instructions}"),
])

prompt_values = {
    "request": "Three words that rhyme with 'cat'",
    "format_instructions": output_parser.get_format_instructions()
}
print(template.format(**prompt_values))

llm_with_prompt = template | llm

response = llm_with_prompt.invoke(input=prompt_values)
parsed_response = output_parser.parse(response.content)
print(parsed_response)

# see the diff when output_parser is used

llm_with_prompt = template | llm | output_parser

print(llm_with_prompt.invoke(input=prompt_values))
