import torch
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

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
            model_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "trust_remote_code": True
            },
            encode_kwargs={"normalize_embeddings": True} # Normalize for cosine similarity
        )

        logger.info("Embedding model loaded successfully.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents.")
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        logger.info("Embedding query.")
        return self.model.embed_query(query)