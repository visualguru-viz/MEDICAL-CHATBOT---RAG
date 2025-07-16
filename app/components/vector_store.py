from langchain_community.vectorstores import FAISS

import os
from app.components.embeddings import get_embedding_model

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)
def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info(f"Loading existing FAISS vector store")
            return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        else:
            logger.info("No existing FAISS vector store found, creating a new one")
    except Exception as e:
        error_message = CustomException("Failed to load FAISS vector store")
        logger.error(str(error_message))
    
def save_vector_store(chunks):
    try:
        if chunks:
            logger.info("text chunks provided to save in FAISS vector store")
        
        logger.info("Generating new vector store")
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(chunks, embedding_model)
        logger.info("Vector store saving")
        db.save_local(DB_FAISS_PATH)

        logger.info("Saving FAISS vector store to disk")
        return db
    
    
    except Exception as e:
        logger.error(f"Error saving FAISS vector store: {e}")
        raise CustomException("Failed to save FAISS vector store") from e