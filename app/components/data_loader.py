import os

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.components.pdf_loader import load_pdf_files, create_text_chunks
from app.components.vector_store import save_vector_store, load_vector_store

from app.config.config import DB_FAISS_PATH

logger = get_logger(__name__)
def process_and_store_pdfs():
    """
    Load data from a PDF file and save it to the vector store.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        None
    """
    try:
        logger.info("Making the vector store")
        documents = load_pdf_files()
        text_chunks = create_text_chunks(documents)
        load_vector_store()
        save_vector_store(text_chunks)
        logger.info("Data loaded and saved successfully")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise CustomException(f"Failed to load data ") from e
    
if __name__ == "__main__":
        process_and_store_pdfs()