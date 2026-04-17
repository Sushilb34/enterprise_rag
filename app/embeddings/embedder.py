import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["HF_HOME"] = "C:/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

        # Determine device — prefer GPU to offload from system RAM
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {self.model_name} on device: {self.device}")

        self.model = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": self.device,
                "trust_remote_code": True,
            },
            encode_kwargs={"normalize_embeddings": True} # Normalize for cosine similarity
        )

        logger.info(f"Embedding model loaded successfully on {self.device}.")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"Embedding {len(texts)} documents.")
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        logger.info("Embedding query.")
        return self.model.embed_query(query)