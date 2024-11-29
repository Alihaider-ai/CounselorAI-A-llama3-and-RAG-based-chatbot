from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
import os
from dotenv import load_dotenv
from typing import Any

class LLMHandler:
    TEMPLATE = """Based solely on the provided context and conversation history, please answer the following question.

    Context:
    {context}

    Conversation History:
    {history}

    Question:
    {question}

    Answer:"""

    def __init__(self, model, max_tokens, temperature):
        load_dotenv()
        self.model_name = model
        self.llm = self._initialize_llm(model, max_tokens, temperature)
        self.prompt = ChatPromptTemplate.from_template(self.TEMPLATE)
        self.rag_chain = None

    def _initialize_llm(self, model, max_tokens, temperature):
        if model.startswith('gemini'):
            return self._setup_gemini(max_tokens, temperature)
        return self._setup_ollama(model, max_tokens, temperature)

    def _setup_gemini(self, max_tokens, temperature):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature
        )

    def _setup_ollama(self, model, max_tokens, temperature):
        return ChatOllama(
            model=model,
            # keep_alive="-1", # Uncomment this line to keep the model alive
            max_tokens=max_tokens,
            temperature=temperature
        )
        
    def detect_constitution_intent(self, question: str,history:Any) -> bool:
        intent_prompt = ChatPromptTemplate.from_template(
            """Determine if the following question is asking about the Constitution of Pakistan.
            Return only 'true' or 'false'.
            
            Question: {question}
            Conversation History: {history}
            Answer (true/false):"""
        )
        
        # Use basic LLM chain for intent detection
        if isinstance(self.llm, ChatOllama):
            response = self.llm.invoke(intent_prompt.format_messages(question=question, history=history))
        else:
            response = self.llm.invoke(intent_prompt.format_messages(question=question, history=history))
        
        return response.content.strip().lower() == 'true'

    def basic_response(self, question: str,history:str) -> str:
        basic_prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that encourages users to ask questions about the Constitution of Pakistan.
            
            If the question is not about the Constitution of Pakistan, provide a brief answer and suggest asking about the Constitution.
            
            User question: {question}
            Conversation History: {history}
            Response:"""
        )
        
        # Use basic LLM chain for response
        if isinstance(self.llm, ChatOllama):
            response = self.llm.invoke(basic_prompt.format_messages(question=question, history=history))
        else:
            response = self.llm.invoke(basic_prompt.format_messages(question=question, history=history))
        
        return response.content.strip()

    def setup_rag_chain(self, retriever):
        self.rag_chain = (
            {"context": retriever, "history": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self._print_and_pass_prompt
        )

    @staticmethod
    def _print_and_pass_prompt(formatted_prompt):
        return formatted_prompt

    def get_answer(self, question, history):
        try:
            rag_context = self.rag_chain.invoke(question)
            print(rag_context)
            formatted_prompt = self.TEMPLATE.format(
                context=rag_context,
                history=history,
                question=question
            )
            return self.llm.invoke(formatted_prompt).content
        except Exception as e:
            raise Exception(f"Error getting answer from {self.model_name}: {str(e)}")