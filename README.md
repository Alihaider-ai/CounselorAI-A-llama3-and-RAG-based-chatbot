# CounselorAI-Knowledge-Based Chatbot for the Constitution of Pakistan

This project implements a Flask-based chatbot that answers questions about the Constitution of Pakistan. It utilizes various components like FAISS for vector search, SentenceTransformer for embeddings, and LLaMA 3 through the Ollama interface for generating responses. The system employs a Retrieval-Augmented Generation (RAG) approach to provide contextually accurate answers.

## Features

- Extracts and processes text from a PDF document of the Constitution of Pakistan.
- Builds and maintains a FAISS index for efficient similarity search.
- Integrates a powerful language model (LLaMA 3) for generating responses.
- Uses a prompt template to format queries for the language model.
- Provides a web interface to interact with the chatbot.

## Components

- **Flask**: Web framework for hosting the chatbot.
- **FAISS**: Library for efficient similarity search.
- **SentenceTransformer**: Model for generating embeddings.
- **PyMuPDF (Fitz)**: Library for extracting text from PDF.
- **Ollama**: Interface for using the LLaMA 3 model.
- **LangChain**: Provides utilities for prompt templates and RAG chaining.

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/Alihaider-ai/CounselorAI-A-llama3-and-RAG-based-chatbot.git
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Prepare the PDF Document**
    Place the `knowledge_base.pdf` (Constitution of Pakistan) in the root directory of the project.

## Usage

1. **Run the Application**
    ```bash
    python app.py
    ```

2. **Access the Chatbot**
    Open your browser and navigate to `http://127.0.0.1:5000` to interact with the chatbot.

## Project Structure

```bash
CounselorAI-A-llama3-and-RAG-based-chatbot/
├── templates/
│   └── index.html            # HTML template for the chatbot interface
│
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── knowledge_base.pdf        # PDF document of the Constitution of Pakistan
├── faiss_index.pkl           # Serialized FAISS index (generated)
├── docs.pkl                  # Serialized document splits (generated)
└── README.md                 # This README file
```

## Key Functions

- **extract_text_from_pdf(pdf_path)**: Extracts text from the specified PDF file.
- **save_faiss_index(db, documents, index_path, docs_path)**: Saves the FAISS index and documents to disk.
- **load_faiss_index(index_path, docs_path)**: Loads the FAISS index and documents from disk.
- **ask_question(question)**: Processes a question using the RAG chain and returns an answer.

## Logging

The application is configured with logging to help debug and monitor its operations. Logs are displayed on the console.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. Please ensure that your contributions align with the overall project goals and code style.



