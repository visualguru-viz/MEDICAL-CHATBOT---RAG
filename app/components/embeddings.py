from langchain_huggingface import HuggingFaceEmbeddings

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger = get_logger(__name__)
def get_embedding_model():
    """
    Get embeddings from HuggingFace model.

    Args:
        model_name (str): Name of the HuggingFace model.

    Returns:
        HuggingFaceEmbeddings: Embeddings object from HuggingFace.
    """
    try:
        logger.info("Loading HuggingFace embeddings model...")
        model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Successfully loaded embeddings model: {model.model_name}")
        return model
    
    except Exception as e:
        logger.error(f"Error getting embeddings for model: {e}")
        raise CustomException(f"Failed to get embeddings for model ") from e