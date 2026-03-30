from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM
    LLM_PROVIDER: str
    OPENAI_API_KEY: str
    LLM_MODEL: str
    LLM_TEMPERATURE: float
    GEMINI_API_KEY: str

    # Embeddings
    EMBEDDING_MODEL: str

    # Chunking
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int

    # Vectorstore
    VECTORSTORE_TYPE: str
    FAISS_INDEX_PATH: str
    BM25_INDEX_PATH: str

    # Retrieval
    TOP_K: int
    HYBRID_ALPHA: float

    # Reranker
    RERANKER_MODEL: str
    RERANK_TOP_K: int

    # Logging
    LOG_LEVEL: str
    LOG_PATH: str

    # Data
    DATA_DIR: str
    EVAL_RAGAS_PATH: str

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()