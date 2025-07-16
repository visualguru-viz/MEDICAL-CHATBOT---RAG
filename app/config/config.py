import os

HF_TOKEN = os.environ.get("HF_TOKEN")

HUGGINGFACE_REPO_ID = "mistralai/Devstral-Small-2507"
DB_FAISS_PATH = "/Users/mahendravarma/LLMOPS/CHATBOT RAG/vector_store/db_faiss"
DATA_PATH = "/Users/mahendravarma/LLMOPS/CHATBOT RAG/app/data/"
# /Users/mahendravarma/LLMOPS/CHATBOT RAG/app/data/
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50