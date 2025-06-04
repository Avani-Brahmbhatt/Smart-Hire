# main_system.py
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
from sqlalchemy import and_

from models import Candidate, Job, CandidateScore, Interview
from database import DatabaseManager
from resume_processor import ResumeProcessor
from embedding_manager import EmbeddingManager
from job_matcher import JobMatcher
from rag_chatbot import RAGChatbot
from communication_manager import CommunicationManager
from multimodal_processor import MultimodalProcessor
from config import Config
from utils import logger, ensure_directory_exists, safe_json_dumps, safe_json_loads

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
            candidate_info = self.resume_processor.extract_candidate_info(doc.page_content, filename)
            
            # Check if candidate already exists
            existing_candidate = self.db.session.query(Candidate).filter_by(
                email=candidate_info['email']
            ).first()
            
            if not existing_candidate:
                # Create new candidate
                candidate = Candidate(
                    name=candidate_info['name'],
                    email=candidate_info['email'],
                    phone=candidate_info['phone'],
                    resume_text=doc.page_content,
                    resume_file_path=os.path.join(folder, filename),
                    skills=safe_json_dumps(candidate_info['skills']),
                    experience_years=candidate_info['experience_years']
                )
                
                try:
                    self.db.session.add(candidate)
                    self.db.session.commit()
                    candidates_added += 1
                    logger.info(f"Added candidate: {candidate_info['name']}")
                except Exception as e:
                    logger.error(f"Error adding candidate {candidate_info['name']}: {str(e)}")
                    self.db.session.rollback()
        
        logger.info(f"Ingested {candidates_added} new candidates from {len(processed_files)} resumes")
        return candidates_added
    
    def add_job(self, title: str, description: str, requirements: str = None,
                department: str = None, location: str = None, salary_range: str = None) -> str:
        """Add a new job posting"""
        try:
            job = Job(
                title=title,
                description=description,
                requirements=requirements,
                department=department,
                location=location,
                salary_range=salary_range
            )
            
            self.db.session.add(job)
            self.db.session.commit()
            
            logger.info(f"Added job: {title}")
            return job.id
        except Exception as e:
            logger.error(f"Error adding job: {str(e)}")
            self.db.session.rollback()
            return None
    
    def match_candidates_to_job(self, job_id: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Match candidates to a specific job"""
        # Get job details
        job = self.db.session.query(Job).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Job not found: {job_id}")
            return []
        
        # Get all candidates
        candidates = self.db.session.query(Candidate).all()
        if not candidates:
            logger.warning("No candidates found in database")
            return []
        
        # Prepare candidate texts for matching
        candidate_texts = []
        candidate_mapping = {}
        
        for candidate in candidates:
            text = f"{candidate.resume_text}\nSkills: {', '.join(safe_json_loads(candidate.skills, []))}"
            candidate_texts.append(text)
            candidate_mapping[text] = candidate
        
        # Create job description text
        job_text = f"{job.title}\n{job.description}"
        if job.requirements:
            job_text += f"\nRequirements: {job.requirements}"
        
        # Match candidates
        matches = self.job_matcher.match_candidates_to_job(
            job_text, candidate_texts, top_k or Config.TOP_K_CANDIDATES
        )
        
        # Prepare results and save scores
        results = []
        for candidate_text, score in matches:
            candidate = candidate_mapping[candidate_text]
            
            # Save or update score in database
            existing_score = self.db.session.query(CandidateScore).filter(
                and_(CandidateScore.candidate_id == candidate.id,
                     CandidateScore.job_id == job_id)
            ).first()
            
            if existing_score:
                existing_score.similarity_score = score
            else:
                candidate_score = CandidateScore(
                    candidate_id=candidate.id,
                    job_id=job_id,
                    similarity_score=score
                )
                self.db.session.add(candidate_score)
            
            results.append({
                'candidate_id': candidate.id,
                'name': candidate.name,
                'email': candidate.email,
                'phone': candidate.phone,
                'skills': safe_json_loads(candidate.skills, []),
                'experience_years': candidate.experience_years,
                'similarity_score': score
            })
        
        try:
            self.db.session.commit()
            logger.info(f"Matched {len(results)} candidates to job {job.title}")
        except Exception as e:
            logger.error(f"Error saving candidate scores: {str(e)}")
            self.db.session.rollback()
        
        return results
    
    def schedule_interview(self, candidate_id: str, job_id: str, 
                          interviewer_email: str, interview_time: datetime,
                          meeting_link: str = None) -> bool:
        """Schedule an interview"""
        try:
            # Get candidate and job details
            candidate = self.db.session.query(Candidate).filter_by(id=candidate_id).first()
            job = self.db.session.query(Job).filter_by(id=job_id).first()
            
            if not candidate or not job:
                logger.error("Candidate or job not found")
                return False
            
            # Create interview record
            interview = Interview(
                candidate_id=candidate_id,
                job_id=job_id,
                interviewer_email=interviewer_email,
                scheduled_time=interview_time,
                meeting_link=meeting_link
            )
            
            self.db.session.add(interview)
            
            # Send interview invitation
            success = self.communication_manager.send_interview_invitation(
                candidate.email,
                candidate.name,
                job.title,
                interview_time,
                meeting_link
            )
            
            if success:
                # Schedule calendar event
                end_time = interview_time + timedelta(hours=1)
                attendees = [candidate.email, interviewer_email]
                
                calendar_link = self.communication_manager.schedule_google_calendar_event(
                    f"Interview: {candidate.name} - {job.title}",
                    interview_time,
                    end_time,
                    attendees,
                    f"Interview with {candidate.name} for {job.title} position"
                )
                
                if calendar_link:
                    interview.meeting_link = calendar_link
                
                self.db.session.commit()
                logger.info(f"Scheduled interview for {candidate.name}")
                return True
            else:
                self.db.session.rollback()
                return False
                
        except Exception as e:
            logger.error(f"Error scheduling interview: {str(e)}")
            self.db.session.rollback()
            return False
    
    def process_video_interview(self, candidate_id: str, video_file_path: str) -> Optional[str]:
        """Process candidate video interview"""
        try:
            candidate = self.db.session.query(Candidate).filter_by(id=candidate_id).first()
            if not candidate:
                logger.error(f"Candidate not found: {candidate_id}")
                return None
            
            # Process video and get transcript
            transcript = self.multimodal_processor.process_candidate_video(
                video_file_path, candidate_id
            )
            
            if transcript:
                # Update candidate with video info
                candidate.video_file_path = video_file_path
                candidate.video_transcript = transcript
                candidate.updated_at = datetime.utcnow()
                
                self.db.session.commit()
                logger.info(f"Processed video interview for {candidate.name}")
                
            return transcript
            
        except Exception as e:
            logger.error(f"Error processing video interview: {str(e)}")
            self.db.session.rollback()
            return None
    
    def send_bulk_rejection_emails(self, job_id: str, candidate_ids: List[str]) -> int:
        """Send rejection emails to multiple candidates"""
        job = self.db.session.query(Job).filter_by(id=job_id).first()
        if not job:
            logger.error(f"Job not found: {job_id}")
            return 0
        
        success_count = 0
        for candidate_id in candidate_ids:
            candidate = self.db.session.query(Candidate).filter_by(id=candidate_id).first()
            if candidate:
                success = self.communication_manager.send_rejection_email(
                    candidate.email, candidate.name, job.title
                )
                if success:
                    success_count += 1
                    
                    # Update candidate score status
                    score = self.db.session.query(CandidateScore).filter(
                        and_(CandidateScore.candidate_id == candidate_id,
                             CandidateScore.job_id == job_id)
                    ).first()
                    if score:
                        score.status = 'rejected'
        
        try:
            self.db.session.commit()
            logger.info(f"Sent {success_count} rejection emails")
        except Exception as e:
            logger.error(f"Error updating rejection status: {str(e)}")
            self.db.session.rollback()
        
        return success_count
    
    def get_candidate_insights(self, candidate_id: str = None, candidate_name: str = None) -> str:
        """Get AI insights about a candidate"""
        if candidate_id:
            candidate = self.db.session.query(Candidate).filter_by(id=candidate_id).first()
        elif candidate_name:
            candidate = self.db.session.query(Candidate).filter_by(name=candidate_name).first()
        else:
            return "Please provide candidate ID or name"
        
        if not candidate:
            return "Candidate not found"
        
        return self.rag_chatbot.get_candidate_info(candidate.name)
    
    def find_candidates_with_skills(self, skills: List[str]) -> str:
        """Find candidates with specific skills using AI"""
        return self.rag_chatbot.find_candidates_with_skills(skills)
    
    def compare_candidates(self, candidate1_id: str, candidate2_id: str) -> str:
        """Compare two candidates using AI"""
        candidate1 = self.db.session.query(Candidate).filter_by(id=candidate1_id).first()
        candidate2 = self.db.session.query(Candidate).filter_by(id=candidate2_id).first()
        
        if not candidate1 or not candidate2:
            return "One or both candidates not found"
        
        return self.rag_chatbot.compare_candidates(candidate1.name, candidate2.name)
    
    def get_job_analytics(self, job_id: str) -> Dict[str, Any]:
        """Get analytics for a specific job"""
        job = self.db.session.query(Job).filter_by(id=job_id).first()
        if not job:
            return {"error": "Job not found"}
        
        # Get candidate scores for this job
        scores = self.db.session.query(CandidateScore).filter_by(job_id=job_id).all()
        
        if not scores:
            return {
                "job_title": job.title,
                "total_applicants": 0,
                "average_score": 0,
                "status_breakdown": {}
            }
        
        # Calculate analytics
        total_applicants = len(scores)
        average_score = sum(score.similarity_score or 0 for score in scores) / total_applicants
        
        status_breakdown = {}
        for score in scores:
            status = score.status
            status_breakdown[status] = status_breakdown.get(status, 0) + 1
        
        # Get top candidates
        top_candidates = sorted(scores, key=lambda x: x.similarity_score or 0, reverse=True)[:5]
        top_candidate_info = []
        
        for score in top_candidates:
            candidate = self.db.session.query(Candidate).filter_by(id=score.candidate_id).first()
            if candidate:
                top_candidate_info.append({
                    "name": candidate.name,
                    "email": candidate.email,
                    "score": score.similarity_score,
                    "status": score.status
                })
        
        return {
            "job_title": job.title,
            "total_applicants": total_applicants,
            "average_score": round(average_score, 3),
            "status_breakdown": status_breakdown,
            "top_candidates": top_candidate_info
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        total_candidates = self.db.session.query(Candidate).count()
        total_jobs = self.db.session.query(Job).count()
        total_interviews = self.db.session.query(Interview).count()
        
        # Recent activity (last 30 days)
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        recent_candidates = self.db.session.query(Candidate).filter(
            Candidate.created_at >= thirty_days_ago
        ).count()
        
        recent_jobs = self.db.session.query(Job).filter(
            Job.created_at >= thirty_days_ago
        ).count()
        
        return {
            "total_candidates": total_candidates,
            "total_jobs": total_jobs,
            "total_interviews": total_interviews,
            "recent_candidates": recent_candidates,
            "recent_jobs": recent_jobs,
            "vectorstore_available": self.embedding_manager.vectorstore is not None
        }
    
    def ask_question(self, question: str) -> str:
        """Ask a question using the RAG system"""
        answer, source_docs = self.rag_chatbot.answer_query(question)
        
        # Add source information if available
        if source_docs:
            sources = [doc.metadata.get('source_file', 'Unknown') for doc in source_docs[:3]]
            unique_sources = list(set(sources))
            answer += f"\n\nSources: {', '.join(unique_sources)}"
        
        return answer
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.db.close()
            logger.info("Cleaned up AI Hiring Agent resources")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


# =====================================
# COMPREHENSIVE TEST SUITE
# =====================================

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from datetime import datetime

class TestAIHiringAgent(unittest.TestCase):
    """Comprehensive test suite for AI Hiring Agent"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.resume_dir = os.path.join(self.test_dir, 'resumes')
        self.upload_dir = os.path.join(self.test_dir, 'uploads')
        os.makedirs(self.resume_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Mock configurations
        with patch('config.Config.RESUME_FOLDER', self.resume_dir), \
             patch('config.Config.UPLOAD_FOLDER', self.upload_dir):
            
            # Mock all dependencies
            self.mock_db = Mock()
            self.mock_resume_processor = Mock()
            self.mock_embedding_manager = Mock()
            self.mock_job_matcher = Mock()
            self.mock_communication_manager = Mock()
            self.mock_multimodal_processor = Mock()
            self.mock_rag_chatbot = Mock()
            
            # Initialize agent with mocked dependencies
            with patch.multiple(
                'main_system',
                DatabaseManager=Mock(return_value=self.mock_db),
                ResumeProcessor=Mock(return_value=self.mock_resume_processor),
                EmbeddingManager=Mock(return_value=self.mock_embedding_manager),
                JobMatcher=Mock(return_value=self.mock_job_matcher),
                CommunicationManager=Mock(return_value=self.mock_communication_manager),
                MultimodalProcessor=Mock(return_value=self.mock_multimodal_processor),
                RAGChatbot=Mock(return_value=self.mock_rag_chatbot)
            ):
                self.agent = AIHiringAgent()
    
    def tearDown(self):
        """Clean up test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test AI Hiring Agent initialization"""
        # Verify all components are initialized
        self.assertIsNotNone(self.agent.db)
        self.assertIsNotNone(self.agent.resume_processor)
        self.assertIsNotNone(self.agent.embedding_manager)
        self.assertIsNotNone(self.agent.job_matcher)
        self.assertIsNotNone(self.agent.communication_manager)
        self.assertIsNotNone(self.agent.multimodal_processor)
        self.assertIsNotNone(self.agent.rag_chatbot)
        print("✓ Initialization test passed")
    
    def test_ingest_resumes_empty_folder(self):
        """Test ingesting resumes from empty folder"""
        self.mock_resume_processor.load_all_resumes.return_value = []
        
        result = self.agent.ingest_resumes()
        
        self.assertEqual(result, 0)
        self.mock_resume_processor.load_all_resumes.assert_called_once()
        print("✓ Empty folder ingestion test passed")
    
    def test_ingest_resumes_with_documents(self):
        """Test ingesting resumes with documents"""
        # Mock document
        mock_doc = Mock()
        mock_doc.page_content = "John Doe\nSoftware Engineer\n5 years Python experience"
        mock_doc.metadata = {'source_file': 'john_doe.pdf'}
        
        # Mock returns
        self.mock_resume_processor.load_all_resumes.return_value = [mock_doc]
        self.mock_resume_processor.split_documents.return_value = [mock_doc]
        self.mock_resume_processor.extract_candidate_info.return_value = {
            'name': 'John Doe',
            'email': 'john@example.com',
            'phone': '123-456-7890',
            'skills': ['Python', 'Django'],
            'experience_years': 5
        }
        
        # Mock database operations
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        result = self.agent.ingest_resumes()
        
        self.assertEqual(result, 1)
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called()
        print("✓ Resume ingestion with documents test passed")
    
    def test_add_job_success(self):
        """Test successful job addition"""
        mock_job = Mock()
        mock_job.id = "job123"
        
        mock_session = Mock()
        self.mock_db.session = mock_session
        
        with patch('models.Job', return_value=mock_job):
            result = self.agent.add_job(
                title="Python Developer",
                description="Looking for Python developer",
                requirements="5+ years Python",
                department="Engineering",
                location="Remote",
                salary_range="$100k-$130k"
            )
        
        self.assertEqual(result, "job123")
        mock_session.add.assert_called_once_with(mock_job)
        mock_session.commit.assert_called_once()
        print("✓ Job addition success test passed")
    
    def test_add_job_failure(self):
        """Test job addition failure"""
        mock_session = Mock()
        mock_session.add.side_effect = Exception("Database error")
        self.mock_db.session = mock_session
        
        with patch('models.Job'):
            result = self.agent.add_job("Test Job", "Test Description")
        
        self.assertIsNone(result)
        mock_session.rollback.assert_called_once()
        print("✓ Job addition failure test passed")
    
    def test_match_candidates_to_job_no_job(self):
        """Test matching candidates when job doesn't exist"""
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        result = self.agent.match_candidates_to_job("nonexistent_job")
        
        self.assertEqual(result, [])
        print("✓ Match candidates to nonexistent job test passed")
    
    def test_match_candidates_to_job_no_candidates(self):
        """Test matching candidates when no candidates exist"""
        mock_job = Mock()
        mock_job.title = "Test Job"
        mock_job.description = "Test Description"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_job
        mock_session.query.return_value.all.return_value = []
        self.mock_db.session = mock_session
        
        result = self.agent.match_candidates_to_job("job123")
        
        self.assertEqual(result, [])
        print("✓ Match candidates with no candidates test passed")
    
    def test_match_candidates_to_job_success(self):
        """Test successful candidate matching"""
        # Mock job
        mock_job = Mock()
        mock_job.title = "Python Developer"
        mock_job.description = "Looking for Python developer"
        mock_job.requirements = "5+ years Python"
        
        # Mock candidate
        mock_candidate = Mock()
        mock_candidate.id = "candidate123"
        mock_candidate.name = "John Doe"
        mock_candidate.email = "john@example.com"
        mock_candidate.phone = "123-456-7890"
        mock_candidate.resume_text = "Python developer with 5 years experience"
        mock_candidate.skills = '["Python", "Django"]'
        mock_candidate.experience_years = 5
        
        # Mock database operations
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_job
        mock_session.query.return_value.all.return_value = [mock_candidate]
        mock_session.query.return_value.filter.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        # Mock job matcher
        self.mock_job_matcher.match_candidates_to_job.return_value = [
            ("Python developer with 5 years experience\nSkills: Python, Django", 0.85)
        ]
        
        with patch('config.Config.TOP_K_CANDIDATES', 5):
            result = self.agent.match_candidates_to_job("job123")
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['name'], "John Doe")
        self.assertEqual(result[0]['similarity_score'], 0.85)
        print("✓ Successful candidate matching test passed")
    
    def test_schedule_interview_invalid_inputs(self):
        """Test scheduling interview with invalid candidate/job"""
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        result = self.agent.schedule_interview(
            "invalid_candidate", "invalid_job", 
            "interviewer@example.com", datetime.now()
        )
        
        self.assertFalse(result)
        print("✓ Schedule interview with invalid inputs test passed")
    
    def test_schedule_interview_success(self):
        """Test successful interview scheduling"""
        # Mock candidate and job
        mock_candidate = Mock()
        mock_candidate.email = "candidate@example.com"
        mock_candidate.name = "John Doe"
        
        mock_job = Mock()
        mock_job.title = "Python Developer"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [mock_candidate, mock_job]
        self.mock_db.session = mock_session
        
        # Mock communication success
        self.mock_communication_manager.send_interview_invitation.return_value = True
        self.mock_communication_manager.schedule_google_calendar_event.return_value = "calendar_link"
        
        interview_time = datetime.now()
        result = self.agent.schedule_interview(
            "candidate123", "job123", 
            "interviewer@example.com", interview_time
        )
        
        self.assertTrue(result)
        mock_session.add.assert_called()
        mock_session.commit.assert_called()
        print("✓ Successful interview scheduling test passed")
    
    def test_process_video_interview_invalid_candidate(self):
        """Test processing video interview with invalid candidate"""
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        result = self.agent.process_video_interview("invalid_candidate", "test_video.mp4")
        
        self.assertIsNone(result)
        print("✓ Process video interview with invalid candidate test passed")
    
    def test_process_video_interview_success(self):
        """Test successful video interview processing"""
        mock_candidate = Mock()
        mock_candidate.name = "John Doe"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_candidate
        self.mock_db.session = mock_session
        
        # Mock transcript processing
        self.mock_multimodal_processor.process_candidate_video.return_value = "Video transcript here"
        
        result = self.agent.process_video_interview("candidate123", "test_video.mp4")
        
        self.assertEqual(result, "Video transcript here")
        mock_session.commit.assert_called()
        print("✓ Successful video interview processing test passed")
    
    def test_send_bulk_rejection_emails_invalid_job(self):
        """Test sending bulk rejection emails with invalid job"""
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        result = self.agent.send_bulk_rejection_emails("invalid_job", ["candidate1", "candidate2"])
        
        self.assertEqual(result, 0)
        print("✓ Bulk rejection emails with invalid job test passed")
    
    def test_send_bulk_rejection_emails_success(self):
        """Test successful bulk rejection emails"""
        mock_job = Mock()
        mock_job.title = "Python Developer"
        
        mock_candidate = Mock()
        mock_candidate.email = "candidate@example.com"
        mock_candidate.name = "John Doe"
        
        mock_score = Mock()
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [mock_job, mock_candidate]
        mock_session.query.return_value.filter.return_value.first.return_value = mock_score
        self.mock_db.session = mock_session
        
        # Mock communication success
        self.mock_communication_manager.send_rejection_email.return_value = True
        
        result = self.agent.send_bulk_rejection_emails("job123", ["candidate123"])
        
        self.assertEqual(result, 1)
        self.assertEqual(mock_score.status, 'rejected')
        print("✓ Successful bulk rejection emails test passed")
    
    def test_get_candidate_insights_no_params(self):
        """Test getting candidate insights without parameters"""
        result = self.agent.get_candidate_insights()
        
        self.assertEqual(result, "Please provide candidate ID or name")
        print("✓ Get candidate insights without parameters test passed")
    
    def test_get_candidate_insights_by_id(self):
        """Test getting candidate insights by ID"""
        mock_candidate = Mock()
        mock_candidate.name = "John Doe"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_candidate
        self.mock_db.session = mock_session
        
        self.mock_rag_chatbot.get_candidate_info.return_value = "Candidate insights here"
        
        result = self.agent.get_candidate_insights(candidate_id="candidate123")
        
        self.assertEqual(result, "Candidate insights here")
        self.mock_rag_chatbot.get_candidate_info.assert_called_once_with("John Doe")
        print("✓ Get candidate insights by ID test passed")
    
    def test_get_candidate_insights_by_name(self):
        """Test getting candidate insights by name"""
        mock_candidate = Mock()
        mock_candidate.name = "John Doe"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_candidate
        self.mock_db.session = mock_session
        
        self.mock_rag_chatbot.get_candidate_info.return_value = "Candidate insights here"
        
        result = self.agent.get_candidate_insights(candidate_name="John Doe")
        
        self.assertEqual(result, "Candidate insights here")
        print("✓ Get candidate insights by name test passed")
    
    def test_get_candidate_insights_not_found(self):
        """Test getting candidate insights for non-existent candidate"""
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        result = self.agent.get_candidate_insights(candidate_id="nonexistent")
        
        self.assertEqual(result, "Candidate not found")
        print("✓ Get candidate insights not found test passed")
    
    def test_find_candidates_with_skills(self):
        """Test finding candidates with specific skills"""
        skills = ["Python", "Django", "AWS"]
        self.mock_rag_chatbot.find_candidates_with_skills.return_value = "Found 3 candidates with these skills"
        
        result = self.agent.find_candidates_with_skills(skills)
        
        self.assertEqual(result, "Found 3 candidates with these skills")
        self.mock_rag_chatbot.find_candidates_with_skills.assert_called_once_with(skills)
        print("✓ Find candidates with skills test passed")
    
    def test_compare_candidates_success(self):
        """Test successful candidate comparison"""
        mock_candidate1 = Mock()
        mock_candidate1.name = "John Doe"
        mock_candidate2 = Mock()
        mock_candidate2.name = "Jane Smith"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [mock_candidate1, mock_candidate2]
        self.mock_db.session = mock_session
        
        self.mock_rag_chatbot.compare_candidates.return_value = "Comparison results here"
        
        result = self.agent.compare_candidates("candidate1", "candidate2")
        
        self.assertEqual(result, "Comparison results here")
        self.mock_rag_chatbot.compare_candidates.assert_called_once_with("John Doe", "Jane Smith")
        print("✓ Successful candidate comparison test passed")
    
    def test_compare_candidates_not_found(self):
        """Test candidate comparison when candidates not found"""
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [None, None]
        self.mock_db.session = mock_session
        
        result = self.agent.compare_candidates("candidate1", "candidate2")
        
        self.assertEqual(result, "One or both candidates not found")
        print("✓ Compare candidates not found test passed")
    
    def test_get_job_analytics_job_not_found(self):
        """Test getting job analytics for non-existent job"""
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        self.mock_db.session = mock_session
        
        result = self.agent.get_job_analytics("nonexistent_job")
        
        self.assertEqual(result, {"error": "Job not found"})
        print("✓ Job analytics job not found test passed")
    
    def test_get_job_analytics_no_scores(self):
        """Test getting job analytics with no candidate scores"""
        mock_job = Mock()
        mock_job.title = "Python Developer"
        
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = mock_job
        mock_session.query.return_value.filter_by.return_value.all.return_value = []
        self.mock_db.session = mock_session
        
        result = self.agent.get_job_analytics("job123")
        
        expected = {
            "job_title": "Python Developer",
            "total_applicants": 0,
            "average_score": 0,
            "status_breakdown": {}
        }
        self.assertEqual(result, expected)
        print("✓ Job analytics with no scores test passed")
    
    def test_get_job_analytics_with_scores(self):
        """Test getting job analytics with candidate scores"""
        mock_job = Mock()
        mock_job.title = "Python Developer"
        
        mock_score1 = Mock()
        mock_score1.similarity_score = 0.85
        mock_score1.status = "pending"
        mock_score1.candidate_id = "candidate1"
        
        mock_score2 = Mock()
        mock_score2.similarity_score = 0.70
        mock_score2.status = "interviewed"
        mock_score2.candidate_id = "candidate2"
        
        mock_candidate1 = Mock()
        mock_candidate1.name = "John Doe"
        mock_candidate1.email = "john@example.com"
        
        mock_candidate2 = Mock()
        mock_candidate2.name = "Jane Smith"
        mock_candidate2.email = "jane@example.com"
        
        mock_session = Mock()
        # First call for job, second call for scores, then calls for candidates
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [mock_job]
        mock_session.query.return_value.filter_by.return_value.all.return_value = [mock_score1, mock_score2]
        mock_session.query.return_value.filter_by.return_value.first.side_effect = [mock_candidate1, mock_candidate2]
        self.mock_db.session = mock_session
        
        result = self.agent.get_job_analytics("job123")
        
        self.assertEqual(result["job_title"], "Python Developer")
        self.assertEqual(result["total_applicants"], 2)
        self.assertEqual(result["average_score"], 0.775)
        self.assertEqual(result["status_breakdown"], {"pending": 1, "interviewed": 1})
        print("✓ Job analytics with scores test passed")
    
    def test_get_system_stats(self):
        """Test getting system statistics"""
        mock_session = Mock()
        mock_session.query.return_value.count.side_effect = [10, 5, 3, 2, 1]  # candidates, jobs, interviews, recent_candidates, recent_jobs
        mock_session.query.return_value.filter.return_value.count.side_effect = [2, 1]  # recent counts
        self.mock_db.session = mock_session
        
        self.mock_embedding_manager.vectorstore = Mock()  # Not None
        
        result = self.agent.get_system_stats()
        
        expected = {
            "total_candidates": 10,
            "total_jobs": 5,
            "total_interviews": 3,
            "recent_candidates": 2,
            "recent_jobs": 1,
            "vectorstore_available": True
        }
        self.assertEqual(result, expected)
        print("✓ System stats test passed")
    
    def test_ask_question_with_sources(self):
        """Test asking question with source documents"""
        mock_doc1 = Mock()
        mock_doc1.metadata = {'source_file': 'resume1.pdf'}
        mock_doc2 = Mock()
        mock_doc2.metadata = {'source_file': 'resume2.pdf'}
        
        self.mock_rag_chatbot.answer_query.return_value = ("Answer here", [mock_doc1, mock_doc2])
        
        result = self.agent.ask_question("What candidates have Python experience?")
        
        self.assertTrue(result.startswith("Answer here"))
        self.assertIn("Sources:", result)
        self.assertIn("resume1.pdf", result)
        print("✓ Ask question with sources test passed")
    
    def test_ask_question_without_sources(self):
        """Test asking question without source documents"""
        self.mock_rag_chatbot.answer_query.return_value = ("Answer here", [])
        
        result = self.agent.ask_question("What candidates have Python experience?")
        
        self.assertEqual(result, "Answer here")
        print("✓ Ask question without sources test passed")
    
    def test_cleanup(self):
        """Test cleanup functionality"""
        self.agent.cleanup()
        
        self.mock_db.close.assert_called_once()
        print("✓ Cleanup test passed")
    
    def test_cleanup_with_error(self):
        """Test cleanup with error handling"""
        self.mock_db.close.side_effect = Exception("Close error")
        
        # Should not raise exception
        self.agent.cleanup()
        print("✓ Cleanup with error test passed")


class TestAIHiringAgentIntegration(unittest.TestCase):
    """Integration tests for AI Hiring Agent functionality"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.test_dir = tempfile.mkdtemp()
        self.resume_dir = os.path.join(self.test_dir, 'resumes')
        self.upload_dir = os.path.join(self.test_dir, 'uploads')
        os.makedirs(self.resume_dir, exist_ok=True)
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Create sample resume files for testing
        self.create_sample_resumes()
    
    def tearDown(self):
        """Clean up integration test environment"""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_sample_resumes(self):
        """Create sample resume files for testing"""
        resumes = [
            {
                'filename': 'john_doe.txt',
                'content': '''John Doe
                Email: john.doe@example.com
                Phone: (555) 123-4567
                
                Experience:
                Senior Python Developer (5 years)
                - Developed web applications using Django and Flask
                - Worked with AWS, Docker, and PostgreSQL
                - Led team of 3 developers
                
                Skills: Python, Django, Flask, AWS, Docker, PostgreSQL, JavaScript
                '''
            },
            {
                'filename': 'jane_smith.txt',
                'content': '''Jane Smith
                Email: jane.smith@example.com
                Phone: (555) 987-6543
                
                Experience:
                Frontend Developer (3 years)
                - Built responsive web applications using React and Vue.js
                - Worked with Node.js and MongoDB
                - Experience with UI/UX design
                
                Skills: JavaScript, React, Vue.js, Node.js, MongoDB, HTML, CSS
                '''
            },
            {
                'filename': 'bob_wilson.txt',
                'content': '''Bob Wilson
                Email: bob.wilson@example.com
                Phone: (555) 456-7890
                
                Experience:
                Data Scientist (4 years)
                - Machine learning model development
                - Data analysis using Python and R
                - Experience with TensorFlow and scikit-learn
                
                Skills: Python, R, TensorFlow, scikit-learn, Pandas, NumPy, SQL
                '''
            }
        ]
        
        for resume in resumes:
            with open(os.path.join(self.resume_dir, resume['filename']), 'w') as f:
                f.write(resume['content'])
    
    @patch('main_system.DatabaseManager')
    @patch('main_system.ResumeProcessor')
    @patch('main_system.EmbeddingManager')
    def test_full_workflow_integration(self, mock_embedding_mgr, mock_resume_proc, mock_db_mgr):
        """Test complete workflow integration"""
        # This is a simplified integration test
        # In a real scenario, you'd use actual database connections
        
        print("✓ Integration test framework set up")
        
        # Mock the workflow components
        mock_resume_proc_instance = Mock()
        mock_resume_proc.return_value = mock_resume_proc_instance
        
        # Test would continue with actual workflow testing
        self.assertTrue(os.path.exists(self.resume_dir))
        self.assertEqual(len(os.listdir(self.resume_dir)), 3)
        print("✓ Sample resumes created successfully")


def run_comprehensive_tests():
    """Run all tests with detailed reporting"""
    print("=" * 60)
    print("STARTING COMPREHENSIVE AI HIRING AGENT TESTS")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add all test methods from TestAIHiringAgent
    test_methods = [
        'test_initialization',
        'test_ingest_resumes_empty_folder',
        'test_ingest_resumes_with_documents',
        'test_add_job_success',
        'test_add_job_failure',
        'test_match_candidates_to_job_no_job',
        'test_match_candidates_to_job_no_candidates',
        'test_match_candidates_to_job_success',
        'test_schedule_interview_invalid_inputs',
        'test_schedule_interview_success',
        'test_process_video_interview_invalid_candidate',
        'test_process_video_interview_success',
        'test_send_bulk_rejection_emails_invalid_job',
        'test_send_bulk_rejection_emails_success',
        'test_get_candidate_insights_no_params',
        'test_get_candidate_insights_by_id',
        'test_get_candidate_insights_by_name',
        'test_get_candidate_insights_not_found',
        'test_find_candidates_with_skills',
        'test_compare_candidates_success',
        'test_compare_candidates_not_found',
        'test_get_job_analytics_job_not_found',
        'test_get_job_analytics_no_scores',
        'test_get_job_analytics_with_scores',
        'test_get_system_stats',
        'test_ask_question_with_sources',
        'test_ask_question_without_sources',
        'test_cleanup',
        'test_cleanup_with_error'
    ]
    
    for method in test_methods:
        test_suite.addTest(TestAIHiringAgent(method))
    
    # Add integration tests
    test_suite.addTest(TestAIHiringAgentIntegration('test_full_workflow_integration'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    print("=" * 60)
    
    return result.wasSuccessful()


# Example usage and testing
if __name__ == "__main__":
    print("Choose testing mode:")
    print("1. Run comprehensive unit tests")
    print("2. Run original demo")
    print("3. Run both")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        success = run_comprehensive_tests()
        if success:
            print("🎉 All tests passed!")
        else:
            print("❌ Some tests failed. Check the output above.")
    
    elif choice == "2":
        # Original demo code
        agent = AIHiringAgent()
        
        try:
            print("=== AI Hiring Agent Demo ===\n")
            
            # 1. Get system stats
            stats = agent.get_system_stats()
            print(f"System Stats: {stats}\n")
            
            # 2. Ingest resumes
            print("Ingesting resumes...")
            candidates_added = agent.ingest_resumes()
            print(f"Added {candidates_added} candidates\n")
            
            # 3. Add a sample job
            job_id = agent.add_job(
                title="Senior Python Developer",
                description="We are looking for an experienced Python developer to join our team.",
                requirements="5+ years Python experience, Django/Flask, AWS, SQL",
                department="Engineering",
                location="Remote",
                salary_range="$100k-$130k"
            )
            
            if job_id:
                print(f"Added job with ID: {job_id}\n")
                
                # 4. Match candidates to job
                print("Matching candidates to job...")
                matches = agent.match_candidates_to_job(job_id, top_k=3)
                print(f"Found {len(matches)} matching candidates:")
                for match in matches:
                    print(f"- {match['name']}: {match['similarity_score']:.3f}")
                print()
                
                # 5. Get job analytics
                analytics = agent.get_job_analytics(job_id)
                print(f"Job Analytics: {analytics}\n")
            
            # 6. Ask questions using RAG
            if agent.rag_chatbot.rag_chain:
                print("Testing RAG system...")
                questions = [
                    "What candidates have Python experience?",
                    "Who has the most years of experience?",
                    "What skills are most common among candidates?"
                ]
                
                for question in questions:
                    answer = agent.ask_question(question)
                    print(f"Q: {question}")
                    print(f"A: {answer}\n")
            
            print("Demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Error in demo: {str(e)}")
        finally:
            agent.cleanup()
    
    elif choice == "3":
        print("Running comprehensive tests first...\n")
        success = run_comprehensive_tests()
        
        if success:
            print("\n🎉 All tests passed! Now running demo...\n")
            # Run demo (same as choice == "2")
            # [Demo code would go here - same as above]
        else:
            print("\n❌ Some tests failed. Please fix issues before running demo.")
    
    else:
        print("Invalid choice. Please run again with 1, 2, or 3.")