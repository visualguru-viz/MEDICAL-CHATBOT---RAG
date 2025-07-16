from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

# from langchain_huggingface import HuggingFaceEndpoint    # ⬅️ new wrapper
# from huggingface_hub import InferenceClient

from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_core.prompts import PromptTemplate


# from app.components.llm import load_llm
from app.components.vector_store import load_vector_store

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import HF_TOKEN, HUGGINGFACE_REPO_ID


CUSTOM_PROMPT_TEMPLATE = """ Answer the following medical question in 2-3 lines maximun using only the information provided in the context. If the answer is not present in the context, say "I don't know".

Context: 
{context}

Question: 
{question}

Answer:
"""


logger = get_logger(__name__)

def set_custom_prompt():
    """
    Set a custom prompt template for the RetrievalQA chain.
    """
    try:
        logger.info("Setting custom prompt template for RetrievalQA chain")
        return PromptTemplate(
            template=CUSTOM_PROMPT_TEMPLATE,
            input_variables=["context", "question"]
        )
    except Exception as e:
        logger.error(f"Failed to set custom prompt template: {e}")
        raise CustomException("Failed to set custom prompt template") from e

llm = HuggingFaceHub(
    repo_id=HUGGINGFACE_REPO_ID,            # any text‑gen model
    huggingfacehub_api_token=HF_TOKEN,
    model_kwargs={
        "temperature": 0.0,
        "max_new_tokens": 512,
    },
)
db = load_vector_store()
retriever = db.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": set_custom_prompt()},       # must return PromptTemplate
    return_source_documents=True,
)

# --- modern LangChain execution API ---
query   = "How does LangChain RetrievalQA work?"
result  = qa_chain.invoke(query)        # ← .invoke(**) is now preferred

print(result["result"])
for doc in result["source_documents"]:
    print(doc.metadata["source"], "→", doc.page_content[:120])