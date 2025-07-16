from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# from langchain_huggingface import HuggingFaceEndpoint    # ⬅️ new wrapper
# from huggingface_hub import InferenceClient


from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID


logger = get_logger(__name__)

CUSTOM_PROMPT_TEMPLATE = """ Answer the following medical question in 2-3 lines maximun using only the information provided in the context. If the answer is not present in the context, say "I don't know".

Context: 
{context}

Question: 
{question}

Answer:
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    
def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        # Load the vector store
        db = load_vector_store()
        if db is None:
            raise CustomException("Vector store is not loaded properly")
        
        llm = load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, hf_token=HF_TOKEN)
        if llm is None:
            raise CustomException("Language model is not loaded properly")
        
       
        # Create the RetrievalQA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs = {'k': 1}),  # Use the vector store as a retriever
            chain_type_kwargs={"prompt": set_custom_prompt()},  # returns PromptTemplate
            return_source_documents=False,                      # get citations back
        )

        logger.info("RetrievalQA chain created successfully")
        return qa_chain
    
    except Exception as e:
        logger.error(f"Failed to create RetrievalQA chain: {e}")
        raise CustomException("Failed to create RetrievalQA chain") from e
    
