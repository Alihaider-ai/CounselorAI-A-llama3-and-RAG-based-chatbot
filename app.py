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

# Define paths
pdf_path = "knowledge_base.pdf"
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

# Set up Ollama language model
local_llm = 'llama3'
llm = ChatOllama(model=local_llm, keep_alive="-1", max_tokens=3000, temperature=0)

# Create prompt template
template = """Answer the question based only on the following context and conversation history:
{context}

# Conversation History:
# {history}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(template)

# Function to print and pass through the formatted prompt
def print_and_pass_prompt(formatted_prompt):
    return formatted_prompt

# Create the RAG chain using LCEL with prompt printing and streaming output
rag_chain = (
    {"context": retriever, "history":RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | print_and_pass_prompt
)
print(rag_chain)

# Conversation history
conversation_history = []

# Function to ask questions
def ask_question(question):
    try:
        # Retrieve context from RAG
        rag_context = rag_chain.invoke(question)
        # print(rag_context)
        
        # Format the conversation history
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history])
        
        # Format the prompt with context, conversation history, and current question
        formatted_prompt = template.format(context=rag_context, history=history, question=question)
        
        # Get answer from LLM
        answer = ""
        for chunk in llm.stream(formatted_prompt):
            answer += chunk.content

        # Update conversation history
        conversation_history.append((question, answer))
        
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
