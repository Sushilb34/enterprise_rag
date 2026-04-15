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
    RAGAS_SLEEP_SECONDS: int
    RAGAS_EMBEDDING_MODEL: str

    # ============================================================
    # Crawl4AI Website Ingestion Settings
    # ============================================================

    # Root website
    CRAWL_BASE_URL: str = "https://quickfoxconsulting.com/"

    # Allowed domains safety (prevents crawling external sites)
    CRAWL_ALLOWED_DOMAINS: list[str] = ["quickfoxconsulting.com"]

    # Crawl limits
    CRAWL_MAX_DEPTH: int = 3
    CRAWL_MAX_PAGES: int = 200

    # Respect robots.txt (legal/ethical crawling)
    CRAWL_RESPECT_ROBOTS_TXT: bool = True

    # URL discovery strategy
    CRAWL_USE_SITEMAP: bool = True
    CRAWL_SITEMAP_URL: str | None = None
    CRAWL_DISCOVER_INTERNAL_LINKS: bool = True

    # Request behaviour
    CRAWL_REQUEST_DELAY: float = 1.5
    CRAWL_TIMEOUT: int = 30

    # Content extraction
    CRAWL_REMOVE_NAV_FOOTER: bool = True
    CRAWL_REMOVE_DUPLICATES: bool = True

    # Export options
    CRAWL_EXPORT_MARKDOWN: bool = True
    CRAWL_EXPORT_PDF: bool = True

    # Safety sleep (prevents rate limits / blocking)
    CRAWL_SLEEP_BETWEEN_PAGES: float = 2.0

    # Local LLM Config
    LOCAL_LLM_MODEL: str
    LOCAL_LLM_API_URL: str
    LOCAL_LLM_MAX_TOKENS: int
    LOCAL_LLM_TEMPERATURE: float
    # Switch between local and cloud LLM
    USE_LOCAL_LLM: bool


    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()