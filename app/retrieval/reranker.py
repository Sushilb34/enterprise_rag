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

        top_docs = []

        for rank, (doc, score) in enumerate(scored_docs[: self.top_k], start=1):
            # Store score and rank in metadata
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = rank

            top_docs.append(doc)

        docs_content = [doc.page_content for doc in top_docs]

        logger.info(
            f"Reranking complete | returned={len(top_docs)} | "
            f"top_rerank_score={top_docs[0].metadata.get('rerank_score', 0.0):.4f} | "
            f"docs_content={docs_content[:100]}"
        )

        return top_docs