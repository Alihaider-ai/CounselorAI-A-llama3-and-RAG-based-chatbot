from flask import Blueprint, request, jsonify, render_template
from app.services import ask_question

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/ask', methods=['POST'])
def ask():
    question = request.form['msg']  # Adjust to match your input field name
    answer = ask_question(question)
    return jsonify({'answer': answer})
