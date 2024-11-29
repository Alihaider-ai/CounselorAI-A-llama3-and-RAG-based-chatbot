from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import logging
import traceback
from app.config import Config
from app.vector_store import VectorStore
from app.llm_handler import LLMHandler

class ChatBot:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config.from_object(Config)
        Session(self.app)
        
        # Initialize components
        self.vector_store = VectorStore(Config.INDEX_PATH, Config.DOCS_PATH)
        self.vector_store.initialize(Config.PDF_PATH)
        
        self.llm_handler = LLMHandler(
            Config.LLM_MODEL,
            Config.MAX_TOKENS,
            Config.TEMPERATURE
        )
        self.llm_handler.setup_rag_chain(self.vector_store.get_retriever())
        
        # Setup routes
        self.setup_routes()
        
    def setup_routes(self):
        self.app.before_request(self._before_request)
        self.app.route('/')(self.index)
        self.app.route('/ask', methods=['POST'])(self.ask)

    def _before_request(self):
        session.permanent = True
        self.app.permanent_session_lifetime = Config.SESSION_LIFETIME

    def index(self):
        return render_template('index.html')

    def ask(self):
        try:
            question = request.form['msg']
            if 'conversation_history' not in session:
                session['conversation_history'] = []
            
            conversation_history = session['conversation_history']
            history = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history])
            
            answer = self.llm_handler.get_answer(question, history)
            
            conversation_history.append((question, answer))
            session['conversation_history'] = conversation_history
            
            return jsonify({'answer': answer})
        except Exception as e:
            error_message = "An error occurred while processing your question."
            logging.error(f"{error_message}\n{traceback.format_exc()}")
            return jsonify({'answer': error_message})

    def run(self):
        self.app.run(debug=True, host='0.0.0.0')

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
    chatbot = ChatBot()
    chatbot.run()