from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import numpy as np

from app.core.logger import get_logger
from app.core.config import get_settings

logger = get_logger()
settings = get_settings()


class BM25Store:
    """
    Enterprise BM25 Sparse Retriever

    Responsibilities:
    - Build sparse keyword index
    - Score documents using BM25
    - Return ranked documents
    """

    def __init__(self):
        self.documents: List[Document] = []
        self.tokenized_corpus = []
        self.bm25 = None

    def _tokenize(self, text: str):
        return text.lower().split()

    def build_index(self, documents: List[Document]):
        """
        Build BM25 index from documents.
        """
        if not documents:
            logger.warning("No documents provided to BM25.")
            return

        logger.info(f"Building BM25 index for {len(documents)} documents.")

        self.documents = documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info("BM25 index built successfully.")

    def search(self, query: str, k: int = None) -> List[Document]:
        """
        Perform keyword-based search.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built yet.")

        if k is None:
            k = settings.TOP_K

        logger.info(f"Performing BM25 search | top_k={k}")

        tokenized_query = self._tokenize(query)

        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]

        return [self.documents[i] for i in top_k_indices]