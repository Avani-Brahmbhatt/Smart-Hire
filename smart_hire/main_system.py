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

# Example usage and testing
if __name__ == "__main__":
    # Initialize the AI Hiring Agent
    agent = AIHiringAgent()
    
    try:
        # Example operations
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
        # Cleanup
        agent.cleanup()