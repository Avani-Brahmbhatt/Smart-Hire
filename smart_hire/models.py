# models.py
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import uuid

Base = declarative_base()

class Candidate(Base):
    __tablename__ = 'candidates'
   
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    phone = Column(String)
    resume_text = Column(Text)
    resume_file_path = Column(String)
    video_file_path = Column(String)
    video_transcript = Column(Text)
    skills = Column(Text)  # JSON string
    experience_years = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Job(Base):
    __tablename__ = 'jobs'
   
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    requirements = Column(Text)
    department = Column(String)
    location = Column(String)
    salary_range = Column(String)
    status = Column(String, default='active')  # active, closed, draft
    created_at = Column(DateTime, default=datetime.utcnow)

class CandidateScore(Base):
    __tablename__ = 'candidate_scores'
   
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id = Column(String, nullable=False)
    job_id = Column(String, nullable=False)
    similarity_score = Column(Float)
    manual_score = Column(Float)
    feedback = Column(Text)
    status = Column(String, default='pending')  # pending, interviewed, rejected, hired
    created_at = Column(DateTime, default=datetime.utcnow)

class Interview(Base):
    __tablename__ = 'interviews'
   
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    candidate_id = Column(String, nullable=False)
    job_id = Column(String, nullable=False)
    interviewer_email = Column(String)
    scheduled_time = Column(DateTime)
    meeting_link = Column(String)
    status = Column(String, default='scheduled')  # scheduled, completed, cancelled
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)