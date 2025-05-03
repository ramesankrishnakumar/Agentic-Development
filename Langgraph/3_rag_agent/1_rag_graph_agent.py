from typing import TypedDict, Literal
from typing_extensions import Final

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.chat_models import init_chat_model
from langchain import hub
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from langgraph.graph import START, END
from langchain_core.runnables.graph import MermaidDrawMethod

load_dotenv()

embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_genai")

docs = [
    Document(
        page_content="Bella Vista is owned by Antonio Rossi, a renowned chef with over 20 years of experience in the culinary industry. He started Bella Vista to bring authentic Italian flavors to the community.",
        metadata={"source": "owner.txt"},
    ),
    Document(
        page_content="Bella Vista offers a range of dishes with prices that cater to various budgets. Appetizers start at $8, main courses range from $15 to $35, and desserts are priced between $6 and $12.",
        metadata={"source": "dishes.txt"},
    ),
    Document(
        page_content="Bella Vista is open from Monday to Sunday. Weekday hours are 11:00 AM to 10:00 PM, while weekend hours are extended from 11:00 AM to 11:00 PM.",
        metadata={"source": "restaurant_info.txt"},
    ),
    Document(
        page_content="Bella Vista offers a variety of menus including a lunch menu, dinner menu, and a special weekend brunch menu. The lunch menu features light Italian fare, the dinner menu offers a more extensive selection of traditional and contemporary dishes, and the brunch menu includes both classic breakfast items and Italian specialties.",
        metadata={"source": "restaurant_info.txt"},
    ),
]

db = Chroma.from_documents(docs, embedding_function)

retriever = db.as_retriever(search_kwargs={"k": 2})


# retriever.invoke("When are the opening hours?")

# help return only the documents not the metadata
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


prompt = hub.pull("rlm/rag-prompt")

rag_chain = prompt | llm


class AgentState(TypedDict):
    messages: list[BaseMessage]
    documents: list[Document]
    question: str
    on_topic: str

class GradeQuestion(BaseModel):
    """Boolean value to check whether a question is releated to the restaurant Bella Vista"""

    score: str = Field(
        description="Question is about restaurant? If yes -> 'Yes' if not -> 'No'"
    )


def is_question_relevant(state: AgentState):
    topic_classifier_system_prompt = """You are a classifier that determines whether a user's question is about one of the following topics:

        1. Information about the owner of Bella Vista, which is Antonio Rossi.
        2. Prices of dishes at Bella Vista (restaurant).
        3. Opening hours of Bella Vista (restaurant).

        If the question IS about any of these topics, respond with 'Yes'. Otherwise, respond with 'No'. Remember, ONLY YES or NO, nothing else in the response!
        """
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", topic_classifier_system_prompt),
            ("human", "User question: {question}"),
        ]
    )
    messages = state["messages"]
    last_message = messages[-1]
    question = last_message.content
    grade_llm = grade_prompt | llm.with_structured_output(GradeQuestion)
    response = grade_llm.invoke(input={ "question" : question })
    state["on_topic"] = "yes" if response.score.lower() == "yes" else "no"
    print(f"on_topic_decision: {state['on_topic']}")
    state["question"] = question
    return state

def on_topic_router(state: AgentState) -> Literal["on_topic", "off_topic"]:
    on_topic = state["on_topic"]
    decision: Final = "on_topic" if on_topic.lower() == "yes" else "off_topic"
    print(f"on_topic_router: {decision}")
    return decision

def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state

def on_topic_response(state):
    documents = state["documents"]
    question = state["question"]
    formatted_docs = format_docs(documents)
    response = rag_chain.invoke(input={"context": formatted_docs, "question": question})
    print(f"on_topic_response: {response.content}")
    state["messages"].append(AIMessage(content=response.content))
    return state

def off_topic_response(state):
    print("off_topic_response")
    state["messages"].append(AIMessage(content="I'm sorry, I can't answer that question."))
    return state




graph = StateGraph(AgentState)
graph.add_node("topic_decision", is_question_relevant)
graph.add_node("retriever", retrieve)
# lessons learned, node key and the state variable must ** NOT ** share the same name
graph.add_node("on_topic_node", on_topic_response)
graph.add_node("off_topic_node", off_topic_response)
# Use named conditional routes
graph.add_conditional_edges(
    "topic_decision", 
    on_topic_router,
    {
        "on_topic": "retriever",       # When router returns "on_topic", go to retriever
        "off_topic": "off_topic_node"  # When router returns "off_topic", go to off_topic_node
    }
)
graph.add_edge("retriever", "on_topic_node")
graph.add_edge(START, "topic_decision")
graph.add_edge("on_topic_node", END)
graph.add_edge("off_topic_node", END)

runnable = graph.compile()

# Generate the Mermaid diagram - the edge labels will appear from our conditional routing
diagram = runnable.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
    output_file_path="rag_non_traditional_agent_graph.png"
)

with open("rag_non_traditional_agent_graph.png", "wb") as f:
    f.write(diagram)

opening_hours = runnable.invoke(input={"messages": [HumanMessage(content="What are the opening hours for Bella Vista?")]})

print(opening_hours)

opening_hours = runnable.invoke(input={"messages": [HumanMessage(content="What happens when a car and a bus collide?")]})

print(opening_hours)




