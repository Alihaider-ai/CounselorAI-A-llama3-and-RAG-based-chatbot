import os
import pickle
import logging
import fitz
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import traceback

pdf_path = "knowledge_base.pdf"
index_path = "faiss_index.pkl"
docs_path = "docs.pkl"

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {pdf_path}\n{traceback.format_exc()}")
        raise e

def save_faiss_index(db, documents, index_path, docs_path):
    try:
        with open(index_path, 'wb') as f:
            pickle.dump(db, f)
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)
    except Exception as e:
        logging.error(f"Failed to save FAISS index and documents\n{traceback.format_exc()}")
        raise e

def load_faiss_index(index_path, docs_path):
    try:
        with open(index_path, 'rb') as f:
            db = pickle.load(f)
        with open(docs_path, 'rb') as f:
            documents = pickle.load(f)
        return db, documents
    except Exception as e:
        logging.error(f"Failed to load FAISS index and documents\n{traceback.format_exc()}")
        raise e

def initialize_faiss():
    if os.path.exists(index_path) and os.path.exists(docs_path):
        db, pages = load_faiss_index(index_path, docs_path)
        logging.info("Loaded existing FAISS index from disk.")
    else:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        db = FAISS.from_documents(pages, OllamaEmbeddings(model="nomic-embed-text", show_progress=True))
        save_faiss_index(db, pages, index_path, docs_path)
        logging.info("Created new FAISS index and saved to disk.")
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return db, retriever

def initialize_llm():
    local_llm = 'llama3'
    llm = ChatOllama(model=local_llm, keep_alive="-1", max_tokens=3000, temperature=0) # add base_url if your ollama is on other device
    return llm

def initialize_rag_chain(retriever):
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
    rag_chain = (
        {"context": retriever, "history": RunnablePassthrough(), "question": RunnablePassthrough()}
        | prompt
    )
    return rag_chain, template
