import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)
def load_pdf_files():
    """
    Load a PDF file and split it into chunks.
    
    :param file_path: Path to the PDF file.
    :return: List of text chunks from the PDF.
    """
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"File not found: {DATA_PATH}")

        logger.info(f"Loading PDF file from {DATA_PATH}")
        
        # loader = PyPDFLoader(DATA_PATH)
        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            logger.warning(f"No PDF documents found in {DATA_PATH}")
        else:
            logger.info(f"Loaded {len(documents)} documents from {DATA_PATH}")

        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files: {e}")
        raise CustomException(f"Failed to load PDF files: {e}")


def create_text_chunks(documents):
    try:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        raise CustomException(f"Failed to split documents: {e}")