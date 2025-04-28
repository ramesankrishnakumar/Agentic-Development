from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, \
    AIMessagePromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

template = ChatPromptTemplate([
    SystemMessagePromptTemplate.from_template("""
    You are a travel assistant, who takes in user preference for a hobby, budget and number of days 
    and gives a travel plan that is personalized.
    Break down the travel plan by each day and list down the place & activity to do that is based on user hobby.
    """),
    HumanMessagePromptTemplate.from_template("""
    Hi!, My hobby is {preference}. I have a budget of ${budget} and I want to travel for {duration} days. 
    Can you suggest a travel plan for me?
    """)

])

llm_with_prompt = template | llm


def travel_idea(preference: str, budget: int, duration: int = 3) -> str:
    """Generate a travel idea based on user preferences, budget, and duration."""

    response = llm_with_prompt.invoke(input={
        "preference": preference,
        "budget": budget,
        "duration": duration
    })
    return response.content


if __name__ == "__main__":
    print(travel_idea("hiking", 1000, 5))
    # printing new line and separator line
    print("\n" + "-" * 50 + "\n")
    print(travel_idea("surfing", 500, 2))
