import os
import shutil # Import shutil for removing directories
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain_chroma.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
# Set your Google API key
# Make sure to set this environment variable or replace it with your actual key
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY" # Uncomment and replace if not set as env var

# Directory containing your PDF files
# Ensure this directory exists and contains valid PDF files
PDF_DIRECTORY = "/Users/kramesan/codebase/BankTransactions/simulated_bank_statements"

# Directory to store the local Chroma database
CHROMA_DB_DIRECTORY = "./chroma_db"

# --- Step 1: Load Documents ---
print(f"Loading documents from {PDF_DIRECTORY}...")
try:
    loader = PyPDFDirectoryLoader(PDF_DIRECTORY)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
except Exception as e:
    print(f"Error loading documents: {e}")
    print("Please ensure the directory exists and contains valid PDF files.")
    exit()

# --- Step 2: Split Documents ---
print("Splitting documents into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, # Size of each text chunk
    chunk_overlap=200 # Overlap between chunks to maintain context
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# --- Step 3: Create Embeddings ---
print("Creating embeddings...")
# Initialize the embedding model
# Using GoogleGenerativeAIEmbeddings requires the GOOGLE_API_KEY environment variable to be set
try:
    # Using a commonly used model for embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    print("Embeddings model initialized.")
except Exception as e:
    print(f"Error initializing embeddings model: {e}")
    print("Please ensure your GOOGLE_API_KEY environment variable is set correctly and the model name is valid.")
    exit()


# --- Step 4: Store Embeddings in a Vector Database (Chroma) ---
print(f"Storing embeddings in Chroma DB at {CHROMA_DB_DIRECTORY}...")

# --- Added: Remove existing Chroma DB directory for a clean start ---
# This helps avoid potential issues with inconsistent database states
if os.path.exists(CHROMA_DB_DIRECTORY):
    print(f"Removing existing Chroma DB directory: {CHROMA_DB_DIRECTORY}")
    shutil.rmtree(CHROMA_DB_DIRECTORY)
    print("Existing Chroma DB directory removed.")

# Now, create a new database
print("Creating a new Chroma DB.")
# Create a new database from the chunks and embeddings
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_DIRECTORY
)
print("New Chroma DB created and populated.")

# # Persist the database to disk (Chroma automatically persists when using persist_directory)
# print("Vector DB persisted to disk.")

# --- Step 5: Set up Retriever ---
# Create a retriever from the vector store
# This retriever will be used to fetch relevant documents based on a query
retriever = vectorstore.as_retriever(search_kwargs={"k": 10}) # Retrieve top 10 most relevant chunks

# --- Step 6: Set up the LLM and Prompt Template ---
print("Setting up LLM and prompt outline_template...")
# Initialize the LLM
# Using GoogleGenerativeAI requires the GOOGLE_API_KEY environment variable to be set
try:
    llm = GoogleGenerativeAI(model="gemini-2.0-flash-001") # You can choose a different model
    print("LLM initialized.")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    print("Please ensure your GOOGLE_API_KEY environment variable is set correctly.")
    exit()


# Define the prompt outline_template
# Modified the prompt to attempt to extract expenses and sum them.
# Note: Relying solely on the LLM for accurate numerical extraction and calculation from unstructured text can be unreliable.
prompt_template = ChatPromptTemplate.from_template("""
Based on the following context, identify all expense transactions for the month(s) mentioned in the question.
List these individual expense transactions with their date, description, and amount.
Then, calculate the total sum of these expenses for the requested month(s).
If a comparison between months is requested, perform the calculation for both months and state the difference or comparison.
Finally, provide a verbose explanation of how you arrived at the answer, including the listed transactions and the calculation steps.

Context:
{context}

Question: {question}

Answer:
""")

# --- Step 7: Create a Langchain Chain ---
# This chain connects the retriever and the LLM
# It first retrieves relevant documents, then passes them to the LLM with the question
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser() # Parse the LLM output to a string
)

# --- Step 8: Ask a Question ---
print("\n--- Ready to answer questions ---")

# Example questions
# To get the sum of expenses for December:
question_dec_sum = "Give me the total sum of all expenses for the month of December. List the individual expenses first."
print(f"\nAsking question: {question_dec_sum}")
answer_dec_sum = rag_chain.invoke(question_dec_sum)
print("\nAnswer:")
print(answer_dec_sum)

# To compare expenses between December and November:
question_compare = "Compare the total expenses for December and January. List the individual expenses for each month first, then provide the total for each month and the difference."
print(f"\nAsking question: {question_compare}")
answer_compare = rag_chain.invoke(question_compare)
print("\nAnswer:")
print(answer_compare)

