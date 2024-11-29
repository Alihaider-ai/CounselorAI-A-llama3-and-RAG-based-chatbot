import fitz
import logging
from langchain_community.document_loaders import PyPDFLoader

class PDFHandler:
    @staticmethod
    def extract_text(pdf_path):
        try:
            doc = fitz.open(pdf_path)
            return " ".join(page.get_text() for page in doc)
        except Exception as e:
            logging.error(f"PDF extraction error: {str(e)}")
            raise
    
    @staticmethod
    def load_and_split(pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            return loader.load_and_split()
        except Exception as e:
            logging.error(f"PDF loading error: {str(e)}")
            raise