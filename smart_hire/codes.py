# job_matcher.py
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from embedding_manager import EmbeddingManager
from config import Config
from utils import logger
from resume_parser import parse_resume  # Assumes you have a resume parser
from job_parser import parse_job_description  # Assumes you have a job parser

model = SentenceTransformer('all-MiniLM-L6-v2')

class JobMatcher:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager

    def skill_match_score(self, resume_skills: List[str], job_skills: List[str]) -> float:
        matched = set(resume_skills).intersection(set(job_skills))
        return len(matched) / len(job_skills) if job_skills else 0.0

    def experience_score(self, actual_years: float, required_years: float) -> float:
        return min(actual_years / required_years, 1.0) if required_years else 1.0

    def education_cert_score(self, resume_edu: str, job_edu: str, resume_certs: List[str], job_certs: List[str]) -> float:
        edu_score = 0.5 if job_edu.lower() in resume_edu.lower() else 0.0
        cert_score = 0.5 if any(cert.lower() in [r.lower() for r in resume_certs] for cert in job_certs) else 0.0
        return edu_score + cert_score

    def semantic_similarity(self, resume_text: str, job_text: str) -> float:
        try:
            resume_vec = model.encode(resume_text, convert_to_tensor=True)
            job_vec = model.encode(job_text, convert_to_tensor=True)
            return util.cos_sim(resume_vec, job_vec).item()
        except Exception as e:
            logger.error(f"Semantic similarity error: {str(e)}")
            return 0.0

    def final_score(self, skill_score: float, experience_score: float,
                    edu_score: float, semantic_score: float,
                    video_score: float = 0.0, coding_score: float = 0.0) -> float:
        return (
            0.3 * skill_score +
            0.2 * experience_score +
            0.1 * edu_score +
            0.4 * semantic_score 
        )

    def match_candidates_to_job(self, job_description: str, candidate_texts: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        top_k = top_k or Config.TOP_K_CANDIDATES
        job_data = parse_job_description(job_description)

        scores = []
        for resume_text in candidate_texts:
            resume_data = parse_resume(resume_text)

            skill_score = self.skill_match_score(resume_data['skills'], job_data['skills'])
            experience_score = self.experience_score(resume_data['years_experience'], job_data['required_experience'])
            edu_score = self.education_cert_score(resume_data['education'], job_data['education'], resume_data['certifications'], job_data['certifications'])
            semantic_score = self.semantic_similarity(resume_text, job_description)

            final = self.final_score(skill_score, experience_score, edu_score, semantic_score)

            if skill_score >= Config.SKILL_THRESHOLD and experience_score >= Config.EXPERIENCE_THRESHOLD:
                scores.append((resume_text, final))

        scores.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Selected top {min(top_k, len(scores))} candidates out of {len(candidate_texts)}")
        return scores[:top_k]

    def score_candidate_for_job(self, candidate_text: str, job_description: str) -> float:
        try:
            resume_data = parse_resume(candidate_text)
            job_data = parse_job_description(job_description)

            skill_score = self.skill_match_score(resume_data['skills'], job_data['skills'])
            experience_score = self.experience_score(resume_data['years_experience'], job_data['required_experience'])
            edu_score = self.education_cert_score(resume_data['education'], job_data['education'], resume_data['certifications'], job_data['certifications'])
            semantic_score = self.semantic_similarity(candidate_text, job_description)

            return self.final_score(skill_score, experience_score, edu_score, semantic_score)
        except Exception as e:
            logger.error(f"Error scoring candidate: {str(e)}")
            return 0.0




# Sample test data
resume_text = """
John Doe is a software engineer with 5 years of experience in Python, Django, and REST APIs.
He holds a B.Tech in Computer Science and is certified in AWS Solutions Architect.
"""

job_description = """
We are hiring a backend engineer with at least 3 years of experience in Python and Django.
Bachelor's degree in Computer Science required. AWS certification is a plus.
"""

# Initialize matcher
embedding_manager = EmbeddingManager()
matcher = JobMatcher(embedding_manager)

# Calculate and print the score
score = matcher.score_candidate_for_job(resume_text, job_description)
print(f"Candidate Match Score: {score:.4f}")