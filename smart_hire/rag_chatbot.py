# rag_chatbot.py
from typing import Tuple, List, Optional
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document
from embedding_manager import EmbeddingManager
from config import Config
from utils import logger

class RAGChatbot:
    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.LLM_MODEL,
            temperature=0.1
        )
        self.rag_chain = None
        self._initialize_chain()
   
    def _initialize_chain(self):
        """Initialize the RAG chain"""
        if not self.embedding_manager.vectorstore:
            logger.warning("No vectorstore available for RAG chain")
            return
       
        try:
            retriever = self.embedding_manager.vectorstore.as_retriever(
                search_kwargs={"k": 5}
            )
            self.rag_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            logger.info("RAG chain initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing RAG chain: {str(e)}")
   
    def answer_query(self, query: str) -> Tuple[str, List[Document]]:
        """Answer a query using RAG"""
        if not self.rag_chain:
            return "RAG system not available. Please ensure documents are loaded.", []
       
        try:
            response = self.rag_chain({"query": query})
            answer = response.get("result", "No answer found.")
            source_docs = response.get("source_documents", [])
           
            logger.info(f"Answered query with {len(source_docs)} source documents")
            return answer, source_docs
           
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return f"Error processing query: {str(e)}", []
   
    def get_candidate_info(self, candidate_name: str) -> str:
        """Get information about a specific candidate"""
        query = f"Tell me about {candidate_name}, their skills, experience, and qualifications."
        answer, _ = self.answer_query(query)
        return answer
   
    def find_candidates_with_skills(self, skills: List[str]) -> str:
        """Find candidates with specific skills"""
        skills_str = ", ".join(skills)
        query = f"Which candidates have experience with {skills_str}?"
        answer, _ = self.answer_query(query)
        return answer
   
    def compare_candidates(self, candidate1: str, candidate2: str) -> str:
        """Compare two candidates"""
        query = f"Compare {candidate1} and {candidate2}. What are their strengths and differences?"
        answer, _ = self.answer_query(query)
        return answer
