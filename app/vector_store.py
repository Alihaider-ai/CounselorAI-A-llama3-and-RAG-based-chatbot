import pickle
import logging
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

class VectorStore:
    def __init__(self, index_path, docs_path):
        self.index_path = index_path
        self.docs_path = docs_path
        self.db = None
        self.documents = None

    def initialize(self, pdf_path):
        try:
            if self._load_existing_index():
                return
            self._create_new_index(pdf_path)
        except Exception as e:
            logging.error(f"Vector store initialization error: {str(e)}")
            raise

    def _load_existing_index(self):
        try:
            with open(self.index_path, 'rb') as f:
                self.db = pickle.load(f)
            with open(self.docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            return True
        except:
            return False

    def _create_new_index(self, pdf_path):
        from pdf_handler import PDFHandler
        pages = PDFHandler.load_and_split(pdf_path)
        embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
        self.db = FAISS.from_documents(pages, embeddings)
        self.documents = pages
        self._save_index()

    def _save_index(self):
        with open(self.index_path, 'wb') as f:
            pickle.dump(self.db, f)
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.documents, f)

    def get_retriever(self):
        return self.db.as_retriever(search_type="similarity", search_kwargs={"k": 5})