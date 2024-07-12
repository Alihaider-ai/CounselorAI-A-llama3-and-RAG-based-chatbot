from flask import Flask, request, jsonify, render_template
import os
import logging
import fitz  # PyMuPDF
import pickle
from sentence_transformers import SentenceTransformer
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Initialize the Flask application
app = Flask(__name__)

# Create embeddings using sentence-transformers
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Define paths
pdf_path = "knowledge base.pdf"
index_path = "faiss_index.pkl"
docs_path = "docs.pkl"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to save FAISS index and documents
def save_faiss_index(db, documents, index_path, docs_path):
    with open(index_path, 'wb') as f:
        pickle.dump(db, f)
    with open(docs_path, 'wb') as f:
        pickle.dump(documents, f)

# Function to load FAISS index and documents
def load_faiss_index(index_path, docs_path):
    with open(index_path, 'rb') as f:
        db = pickle.load(f)
    with open(docs_path, 'rb') as f:
        documents = pickle.load(f)
    return db, documents

# Load or create FAISS index
if os.path.exists(index_path) and os.path.exists(docs_path):
    db, pages = load_faiss_index(index_path, docs_path)
    logging.info("Loaded existing FAISS index from disk.")
else:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    db = FAISS.from_documents(pages, HuggingFaceEmbeddings())
    save_faiss_index(db, pages, index_path, docs_path)
    logging.info("Created new FAISS index and saved to disk.")

# Create retriever with similarity search
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up Ollama language model - Gemma 2
local_llm = 'llama3'
llm = ChatOllama(model=local_llm, keep_alive="3h", max_tokens=1000, temperature=0)

# Create prompt template
template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(template)

# Function to print and pass through the formatted prompt
def print_and_pass_prompt(formatted_prompt):
    return formatted_prompt

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | print_and_pass_prompt
    | llm
)

# Function to ask questions
def ask_question(question):
    try:
        answer = ""
        for chunk in rag_chain.stream(question):
            answer += chunk.content
        return answer
    except Exception as e:
        logging.error("An error occurred while processing the question: %s", e)
        return "An error occurred while processing your question."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['msg']  # Adjust to match your input field name
    answer = ask_question(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
