# job_matcher.py
from typing import List, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embedding_manager import EmbeddingManager
from config import Config
from utils import logger

class JobMatcher:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
   
    def match_candidates_to_job(self, job_description: str, candidate_texts: List[str],
                               top_k: int = None) -> List[Tuple[str, float]]:
        """Match candidates to job based on similarity"""
        if not candidate_texts:
            logger.warning("No candidate texts provided for matching")
            return []
       
        top_k = top_k or Config.TOP_K_CANDIDATES
       
        try:
            # Get job embedding
            job_embedding = self.embedding_manager.embed_text(job_description)
            if not job_embedding:
                logger.error("Failed to get job embedding")
                return []
           
            # Get candidate embeddings
            candidate_embeddings = []
            for text in candidate_texts:
                emb = self.embedding_manager.embed_text(text)
                if emb:
                    candidate_embeddings.append(emb)
                else:
                    candidate_embeddings.append([0] * len(job_embedding))
           
            if not candidate_embeddings:
                logger.error("Failed to get candidate embeddings")
                return []
           
            # Calculate similarities
            similarities = cosine_similarity(
                [job_embedding],
                candidate_embeddings
            )[0]
           
            # Get top matches
            top_indices = similarities.argsort()[::-1][:top_k]
            matches = [
                (candidate_texts[i], float(similarities[i]))
                for i in top_indices
                if similarities[i] >= Config.SIMILARITY_THRESHOLD
            ]
           
            logger.info(f"Found {len(matches)} candidate matches above threshold")
            return matches
           
        except Exception as e:
            logger.error(f"Error in candidate matching: {str(e)}")
            return []
   
    def score_candidate_for_job(self, candidate_text: str, job_description: str) -> float:
        """Score a single candidate for a job"""
        try:
            candidate_emb = self.embedding_manager.embed_text(candidate_text)
            job_emb = self.embedding_manager.embed_text(job_description)
           
            if not candidate_emb or not job_emb:
                return 0.0
           
            similarity = cosine_similarity([candidate_emb], [job_emb])[0][0]
            return float(similarity)
           
        except Exception as e:
            logger.error(f"Error scoring candidate: {str(e)}")
            return 0.0
