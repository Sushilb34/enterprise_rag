from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import numpy as np
import pickle
from pathlib import Path

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
        self.index_path = Path(settings.BM25_INDEX_PATH)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def _tokenize(self, text: str):
        return text.lower().split()

    def build_index(self, documents: List[Document]):
        """
        Build BM25 index from documents or load from disk if unchanged.
        """
        if not documents:
            logger.warning("No documents provided to BM25.")
            return

        # Check if we can load from disk
        if self._can_load_from_disk(documents):
            logger.info("Loading BM25 index from disk.")
            self._load_from_disk()
            logger.info("BM25 index loaded successfully.")
            return

        logger.info(f"Building BM25 index for {len(documents)} documents.")

        self.documents = documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self._save_to_disk()

        logger.info("BM25 index built successfully.")

    def search(self, query: str, k: int = None):
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

        return [(self.documents[i], scores[i]) for i in top_k_indices]

    def _can_load_from_disk(self, documents: List[Document]) -> bool:
        """
        Check if index exists on disk and document count matches.
        """
        if not self.index_path.exists():
            return False

        try:
            metadata_path = self.index_path.parent / f"{self.index_path.name}.meta"
            if not metadata_path.exists():
                return False

            with open(metadata_path, "r") as f:
                stored_doc_count = int(f.read().strip())

            return stored_doc_count == len(documents)
        except Exception as e:
            logger.warning(f"Error checking disk index: {e}")
            return False

    def _save_to_disk(self):
        """
        Save BM25 index and documents to disk.
        """
        try:
            # Save the BM25 index and documents
            with open(self.index_path, "wb") as f:
                pickle.dump(
                    {
                        "bm25": self.bm25,
                        "documents": self.documents,
                        "tokenized_corpus": self.tokenized_corpus,
                    },
                    f,
                )

            # Save metadata (document count)
            metadata_path = self.index_path.parent / f"{self.index_path.name}.meta"
            with open(metadata_path, "w") as f:
                f.write(str(len(self.documents)))

            logger.info(f"BM25 index saved to {self.index_path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")

    def _load_from_disk(self):
        """
        Load BM25 index and documents from disk.
        """
        try:
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.documents = data["documents"]
                self.tokenized_corpus = data["tokenized_corpus"]
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            raise

    def is_empty(self) -> bool:
        """
        Check if the BM25 index contains any documents.
        """
        return self.bm25 is None or len(self.documents) == 0