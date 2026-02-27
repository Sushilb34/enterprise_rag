from typing import List
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger()
settings = get_settings()


class EmbeddingModel:
    """
    Enterprise Embedding Layer

    Responsibilities:
    - Load embedding model from config
    - Provide document embedding
    - Provide query embedding
    - Centralize embedding logic
    """

    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL

        logger.info(f"Loading embedding model: {self.model_name}")

        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},  # Change to 'cuda' if GPU available
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Embedding model loaded successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents.")
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        logger.info("Embedding query.")
        return self.model.embed_query(query)