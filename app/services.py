import traceback
import logging
from flask import session
from .utils import initialize_faiss, initialize_llm, initialize_rag_chain

# Initialize components
db, retriever = initialize_faiss()
llm = initialize_llm()
rag_chain, template = initialize_rag_chain(retriever)

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

        # Create the formatted prompt
        formatted_prompt = template.format(context=rag_context, history=history, question=question)

        # Get answer from LLM
        answer = llm.invoke(formatted_prompt).content

        # Update conversation history
        conversation_history.append((question, answer))
        session['conversation_history'] = conversation_history

        return answer
    except Exception as e:
        error_message = "An error occurred while processing your question."
        logging.error(f"{error_message}\n{traceback.format_exc()}")
        return error_message
