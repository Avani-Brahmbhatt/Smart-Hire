# resume_processor.py
import os
from typing import List, Tuple
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import Config
from utils import logger, extract_email_from_text, extract_phone_from_text, extract_name_from_filename
import re

class ResumeProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
   
    def load_document(self, file_path: str) -> List[Document]:
        """Load a single document"""
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding="utf-8")
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return []
           
            documents = loader.load()
            logger.info(f"Loaded document: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
            return []
   
    def load_all_resumes(self, folder_path: str) -> List[Document]:
        """Load all resume documents from folder"""
        documents = []
        if not os.path.exists(folder_path):
            logger.warning(f"Resume folder does not exist: {folder_path}")
            return documents
       
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                docs = self.load_document(file_path)
                # Add metadata
                for doc in docs:
                    doc.metadata['source_file'] = filename
                    doc.metadata['candidate_name'] = extract_name_from_filename(filename)
                documents.extend(docs)
       
        logger.info(f"Loaded {len(documents)} resume documents")
        return documents
   
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        return chunks
   
    def extract_candidate_info(self, text: str, filename: str) -> dict:
        """Extract structured information from resume text"""
        info = {
            'name': extract_name_from_filename(filename),
            'email': extract_email_from_text(text),
            'phone': extract_phone_from_text(text),
            'skills': self.extract_skills(text),
            'experience_years': self.extract_experience_years(text)
        }
        return info
   
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text"""
        # Common technical skills
        skill_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|React|Angular|Vue|Node\.js|Django|Flask)\b',
            r'\b(?:SQL|MySQL|PostgreSQL|MongoDB|Redis|Elasticsearch)\b',
            r'\b(?:AWS|Azure|GCP|Docker|Kubernetes|Jenkins)\b',
            r'\b(?:Machine Learning|ML|AI|Deep Learning|TensorFlow|PyTorch)\b',
            r'\b(?:HTML|CSS|Bootstrap|Tailwind)\b'
        ]
       
        skills = []
        for pattern in skill_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend([match for match in matches if match not in skills])
       
        return skills[:10]  # Limit to top 10 skills
   
    def extract_experience_years(self, text: str) -> float:
        """Extract years of experience from resume text"""
        # Look for patterns like "5 years", "2+ years", etc.
        experience_patterns = [
            r'(\d+)[\+\-\s]*years?\s+(?:of\s+)?experience',
            r'(\d+)[\+\-\s]*years?\s+in',
            r'(\d+)[\+\-\s]*yrs?\s+(?:of\s+)?experience'
        ]
       
        years = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            years.extend([int(match) for match in matches])
       
        return max(years) if years else 0.0
