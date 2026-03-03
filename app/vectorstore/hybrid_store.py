from typing import List, Dict
import numpy as np
from collections import defaultdict

from langchain_core.documents import Document

from app.vectorstore.faiss_store import FAISSVectorStore
from app.vectorstore.bm25_store import BM25Store
from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger()
settings = get_settings()


class HybridRetriever:
    """
    Enterprise Hybrid Retriever

    Combines:
    - Dense vector search (FAISS)
    - Sparse keyword search (BM25)
    """

    def __init__(self, documents: List[Document]):
        self.alpha = settings.HYBRID_ALPHA  # weight for vector score
        self.top_k = settings.TOP_K

        logger.info(f"Initializing Hybrid Retriever | alpha={self.alpha}")

        # Initialize vector store
        self.vector_store = FAISSVectorStore()

        # Initialize BM25 store
        self.bm25_store = BM25Store()
        self.bm25_store.build_index(documents)

    def _normalize(self, scores: List[float]) -> List[float]:
        """
        Min-max normalization
        """
        if not scores:
            return []

        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score == 0:
            return [1.0 for _ in scores]

        return [(s - min_score) / (max_score - min_score) for s in scores]

    def retrieve(self, query: str) -> List[Document]:
        logger.info("Running hybrid retrieval...")

        # Vector search with scores
        vector_results = self.vector_store.vectorstore.similarity_search_with_score(
            query, k=self.top_k
        )

        vector_docs = [doc for doc, score in vector_results]
        vector_scores = [score for doc, score in vector_results]

        # Convert FAISS distance to similarity
        vector_scores = [-score for score in vector_scores]
        #FAISS returns distance
        #Distance means smaller = better
        #Hybrid scoring needs larger scores = better
        #So we invert it by negating the scores

        # BM25 search
        bm25_docs = self.bm25_store.search(query, k=self.top_k)

        # Fake BM25 scoring (rank-based)
        bm25_scores = list(reversed(range(1, len(bm25_docs) + 1)))

        # Normalize scores
        norm_vector = self._normalize(vector_scores)
        norm_bm25 = self._normalize(bm25_scores)

        combined_scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        # Combine vector scores
        for doc, score in zip(vector_docs, norm_vector):
            doc_id = self._doc_identifier(doc)
            combined_scores[doc_id] += self.alpha * score
            doc_map[doc_id] = doc

        # Combine BM25 scores
        for doc, score in zip(bm25_docs, norm_bm25):
            doc_id = self._doc_identifier(doc)
            combined_scores[doc_id] += (1 - self.alpha) * score
            doc_map[doc_id] = doc

        # Sort final results
        ranked_docs = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )

        final_docs = [doc_map[doc_id] for doc_id, _ in ranked_docs[: self.top_k]]

        logger.info(f"Hybrid retrieval complete | returned={len(final_docs)}")

        return final_docs

    def _doc_identifier(self, doc: Document) -> str:
        """
        Unique identifier for deduplication.
        """
        return (
            f"{doc.metadata.get('file_name')}_"
            f"{doc.metadata.get('page_number')}_"
            f"{doc.metadata.get('chunk_id')}"
        )