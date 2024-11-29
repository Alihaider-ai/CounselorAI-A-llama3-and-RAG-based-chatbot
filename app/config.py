import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.urandom(24)
    SESSION_TYPE = 'filesystem'
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    TIMEOUT = 600
    SESSION_LIFETIME = timedelta(minutes=60)
    
    # Paths
    PDF_PATH = "knowledge_base.pdf"
    INDEX_PATH = "faiss_index.pkl"
    DOCS_PATH = "docs.pkl"
    
    # LLM Config
    LLM_MODEL = "gemini-pro"
    MAX_TOKENS = 3000
    TEMPERATURE = 0