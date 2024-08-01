from flask import Flask, request, jsonify, render_template, session
import os
import logging
import traceback
import fitz  # PyMuPDF
import pickle
from datetime import timedelta
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from flask_session import Session
# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Replace with a secure key in production
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['TIMEOUT'] = 600
Session(app)
# Define paths
pdf_path = "knowledge_base.pdf"
index_path = "faiss_index.pkl"
docs_path = "docs.pkl"

# handeling time out in before request to make sure we dont run into timeout errors
@app.before_request
def before_request():
    session.permanent = True
    app.permanent_session_lifetime = timedelta(minutes=60)

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
    db = FAISS.from_documents(pages, OllamaEmbeddings(model="nomic-embed-text",show_progress=True))
    save_faiss_index(db, pages, index_path, docs_path)
    logging.info("Created new FAISS index and saved to disk.")

# Create retriever with similarity search
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Set up Ollama language model
local_llm = 'llama3'
llm = ChatOllama(model=local_llm, keep_alive="-1", max_tokens=3000, temperature=0) # add base_url if your model is deployed on other pc

# Create prompt template
template = """Based solely on the provided context and conversation history, please answer the following question.

Context:
{context}

Conversation History:
{history}

Question:
{question}

Answer:

If the user engages in small talk instead of asking a question ignore the context, continue the conversation naturally and when question is asked responed accoriding to the context provided Maintain a professional tone throughout."""
prompt = ChatPromptTemplate.from_template(template)

# Function to print and pass through the formatted prompt
def print_and_pass_prompt(formatted_prompt):
    return formatted_prompt

# Create the RAG chain
rag_chain = (
    {"context": retriever, "history": RunnablePassthrough(), "question": RunnablePassthrough()}
    | prompt
    | print_and_pass_prompt
)
print(rag_chain)

# Function to ask questions
def ask_question(question):
    try:
        # Retrieve context from RAG
        rag_context = rag_chain.invoke(question)
        
        # Retrieve or initialize conversation history
        if 'conversation_history' not in session:
            session['conversation_history'] = []
        conversation_history = session['conversation_history']

        # Format the conversation history
        history = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history])
        
        # Format the prompt with context, conversation history, and current question
        formatted_prompt = template.format(context=rag_context, history=history, question=question)
        
        # Get answer from LLM
        answer = llm.invoke(formatted_prompt).content
        print("this is the answer", answer)
        # for chunk in llm.stream(formatted_prompt):
        #     answer += chunk.content

        # Update conversation history
        conversation_history.append((question, answer))
        session['conversation_history'] = conversation_history
        
        return answer
    except Exception as e:
        error_message = "An error occurred while processing your question."
        logging.error(f"{error_message}\n{traceback.format_exc()}")
        return error_message

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['msg']  # Adjust to match your input field name
    answer = ask_question(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')
