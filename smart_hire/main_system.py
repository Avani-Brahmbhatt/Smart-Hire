
# main_system.py
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from models import Candidate, Job, CandidateScore, Interview
from database import DatabaseManager
from resume_processor import ResumeProcessor
from embedding_manager import EmbeddingManager
from job_matcher import JobMatcher
from rag_chatbot import RAGChatbot
from communication_manager import CommunicationManager
from multimodal_processor import MultimodalProcessor
from config import Config
from utils import logger, ensure_directory_exists, safe_json_dumps

class AIHiringAgent:
    def __init__(self):
        # Initialize components
        self.db = DatabaseManager()
        self.resume_processor = ResumeProcessor()
        self.embedding_manager = EmbeddingManager()
        self.job_matcher = JobMatcher(self.embedding_manager)
        self.communication_manager = CommunicationManager()
        self.multimodal_processor = MultimodalProcessor()
       
        # Initialize directories
        ensure_directory_exists(Config.RESUME_FOLDER)
        ensure_directory_exists(Config.UPLOAD_FOLDER)
       
        # Load or create vectorstore
        self._initialize_vectorstore()
       
        # Initialize RAG chatbot
        self.rag_chatbot = RAGChatbot(self.embedding_manager)
       
        logger.info("AI Hiring Agent initialized successfully")
   
    def _initialize_vectorstore(self):
        """Initialize vectorstore with existing resumes"""
        # Try to load existing vectorstore
        if not self.embedding_manager.load_vectorstore():
            # Create new vectorstore from existing resumes
            self.ingest_resumes()
   
    def ingest_resumes(self, resume_folder: str = None) -> int:
        """Ingest resumes from folder"""
        folder = resume_folder or Config.RESUME_FOLDER
       
        # Load and process resumes
        documents = self.resume_processor.load_all_resumes(folder)
        if not documents:
            logger.warning("No resumes found to ingest")
            return 0
       
        # Split documents
        chunks = self.resume_processor.split_documents(documents)
       
        # Create or update vectorstore
        if self.embedding_manager.vectorstore:
            self.embedding_manager.add_documents(chunks)
        else:
            self.embedding_manager.create_vectorstore(chunks)
       
        # Process individual resumes and save to database
        processed_files = set()
        candidates_added = 0
       
        for doc in documents:
            filename = doc.metadata.get('source_file', '')
            if filename in processed_files:
                continue
           
            processed_files.add(filename)
           
            # Extract candidate information
            candidate_info = self.resume_