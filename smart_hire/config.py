import os
from typing import Dict, Any

class Config:
    """Configuration management for the AI Hiring Agent"""
   
    # Directories
    RESUME_FOLDER = os.getenv("RESUME_FOLDER", "resumes")
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///hiring_agent.db")
   
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
    GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
    GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
   
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gemma2-9b-it"
    WHISPER_MODEL = "base"
   
    # Processing settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_CANDIDATES = 5
    SIMILARITY_THRESHOLD = 0.3









