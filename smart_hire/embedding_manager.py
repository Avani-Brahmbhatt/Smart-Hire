# embedding_manager.py
from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from config import Config
from utils import logger, ensure_directory_exists

class EmbeddingManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL
        )
        self.vectorstore: Optional[FAISS] = None
        ensure_directory_exists(Config.FAISS_INDEX_DIR)
   
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create new vectorstore from documents"""
        if not documents:
            logger.warning("No documents provided for vectorstore creation")
            return None
       
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
            self.save_vectorstore()
            logger.info(f"Created vectorstore with {len(documents)} documents")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error creating vectorstore: {str(e)}")
            return None
   
    def load_vectorstore(self) -> Optional[FAISS]:
        """Load existing vectorstore"""
        try:
            self.vectorstore = FAISS.load_local(
                Config.FAISS_INDEX_DIR,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info("Loaded existing vectorstore")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"Could not load vectorstore: {str(e)}")
            return None
   
    def save_vectorstore(self):
        """Save vectorstore to disk"""
        if self.vectorstore:
            try:
                self.vectorstore.save_local(Config.FAISS_INDEX_DIR)
                logger.info("Saved vectorstore to disk")
            except Exception as e:
                logger.error(f"Error saving vectorstore: {str(e)}")
   
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vectorstore"""
        if not self.vectorstore:
            logger.warning("No vectorstore loaded. Creating new one.")
            self.create_vectorstore(documents)
            return
       
        try:
            self.vectorstore.add_documents(documents)
            self.save_vectorstore()
            logger.info(f"Added {len(documents)} documents to vectorstore")
        except Exception as e:
            logger.error(f"Error adding documents to vectorstore: {str(e)}")
   
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vectorstore:
            logger.warning("No vectorstore available for search")
            return []
       
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
   
    def embed_text(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            return self.embeddings.embed_query(text)
        except Exception as e:
            logger.error(f"Error embedding text: {str(e)}")
            return []
