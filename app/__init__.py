from flask import Flask
from flask_session import Session
import os
from app.config.logging_config import configure_logging

def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(24)  # Replace with a secure key in production
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['TIMEOUT'] = 600
    
    Session(app)
    
    # Initialize logging
    configure_logging()

    # Import and register Blueprints
    from app.routes import main_bp
    app.register_blueprint(main_bp)

    return app
