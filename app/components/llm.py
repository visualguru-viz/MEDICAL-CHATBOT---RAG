from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from langchain_huggingface import HuggingFaceEndpoint

from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID

logger = get_logger(__name__)

def load_llm(huggingface_repo_id: str = HUGGINGFACE_REPO_ID, hf_token: str = HF_TOKEN):
    try:
        logger.info(f"Loading LLM from Hugging Face Hub: {huggingface_repo_id}")    
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            huggingfacehub_api_token = hf_token,
            temperature=0.3,
            max_new_tokens=256,
            return_full_text=False,
        )
        logger.info(f"Successfully loaded LLM from Hugging Face Hub: {huggingface_repo_id}")
        return llm
    except Exception as e:
        logger.error(f"Failed to load LLM from Hugging Face Hub: {e}")