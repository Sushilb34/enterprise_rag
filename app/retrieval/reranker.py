from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger()
settings = get_settings()


class CrossEncoderReranker:
    """
    Enterprise Cross-Encoder Reranker

    Responsibilities:
    - Load reranker model from config
    - Score (query, document) pairs
    - Return top reranked documents
    """

    def __init__(self):
        self.model_name = settings.RERANKER_MODEL
        self.top_k = settings.RERANK_TOP_K

        logger.info(f"Loading reranker model: {self.model_name}")

        self.model = CrossEncoder(
            self.model_name,
            device="cpu"  # change to "cuda" if GPU available
        )

        logger.info("Reranker model loaded successfully.")

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents using cross-encoder.
        """

        if not documents:
            logger.warning("No documents provided for reranking.")
            return []

        logger.info(f"Reranking {len(documents)} documents...")

        pairs = [(query, doc.page_content) for doc in documents]

        scores = self.model.predict(pairs)

        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        top_docs = [doc for doc, _ in scored_docs[: self.top_k]]

        logger.info(f"Reranking complete | returned={len(top_docs)}")

        return top_docs