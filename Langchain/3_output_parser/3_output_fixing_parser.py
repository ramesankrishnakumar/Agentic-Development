from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv()
llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")
fixing_parser = OutputFixingParser
output_parser = DatetimeOutputParser()

print(output_parser.get_format_instructions())

template = ChatPromptTemplate([
    HumanMessagePromptTemplate.from_template("{request}\n{format_instructions}"),
])

print("\n" + "-" * 50 + "\n")

prompt_values = {
    "request": "What date was the 13th Amendment ratified in India?",
    "format_instructions": output_parser.get_format_instructions()
}

print(template.format(**prompt_values))

print("\n" + "-" * 50 + "\n")

llm_with_prompt = template | llm | output_parser

try:
    response = llm_with_prompt.invoke(input=prompt_values)
    print(response)
except Exception as e:
    # print(e)
    print("-" * 10 + "parsing failed" + "-" * 10)
    try:
        llm_with_prompt = template | llm | fixing_parser.from_llm(llm, output_parser)
        response = llm_with_prompt.invoke(input=prompt_values)
        print("-" * 10 + "parsing fixed" + "-" * 10)
        print(response)
    except Exception as e:
        pass



