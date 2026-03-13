from pathlib import Path
from typing import List, Optional

import faiss
# used for vector store persistence when building new indexes
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.logger import get_logger
from app.embeddings.embedder import EmbeddingModel

logger = get_logger()
settings = get_settings()


class FAISSVectorStore:
    """
    Enterprise FAISS Vector Store with HNSW index + persistence.

    Responsibilities:
    - Create HNSW index
    - Save/load index from disk
    - Add documents
    - Perform similarity search
    """

    def __init__(self):
        self.index_path = Path(settings.FAISS_INDEX_PATH)
        self.embedding_model = EmbeddingModel()
        self.vectorstore: Optional[FAISS] = None

        # Make sure directory exists
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize index safely
        self._load_or_initialize()

    def _create_hnsw_index(self, dimension: int):
        """
        Create FAISS HNSW index.
        """
        logger.info(f"Creating HNSW index | dimension={dimension}")

        index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
        index.hnsw.efConstruction = 200
        index.hnsw.efSearch = 64

        return index

    def _load_or_initialize(self):
        """
        Load existing FAISS index if present,
        otherwise initialize a new one safely.
        """
        try:
            if self.index_path.exists():
                logger.info(f"Loading existing FAISS index from {self.index_path}")
                self.vectorstore = FAISS.load_local(
                    str(self.index_path),
                    self.embedding_model.model,
                    allow_dangerous_deserialization=True,
                )
                logger.info("FAISS index loaded successfully.")
            else:
                logger.info("No existing FAISS index found. Initializing new FAISS store.")

                # Dummy vector to determine embedding dimension
                dummy_vector = self.embedding_model.embed_query("dimension test")
                dimension = len(dummy_vector)

                # Create HNSW index
                index = self._create_hnsw_index(dimension)

                # Create new FAISS store with in-memory docstore
                self.vectorstore = FAISS(
                    embedding_function=self.embedding_model.model,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={},
                )

        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            logger.info("Initializing a new empty FAISS index.")
            dummy_vector = self.embedding_model.embed_query("dimension test")
            dimension = len(dummy_vector)
            index = self._create_hnsw_index(dimension)

            self.vectorstore = FAISS(
                embedding_function=self.embedding_model.model,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )

    def add_documents(self, documents: List[Document]):
        """
        Add documents to FAISS index.
        """
        if not documents:
            logger.warning("No documents provided to add.")
            return

        logger.info(f"Adding {len(documents)} documents to FAISS index.")
        self.vectorstore.add_documents(documents)

        self.save()

    def save(self):
        """
        Persist FAISS index to disk.
        """
        if self.vectorstore is None:
            logger.warning("No FAISS vectorstore to save.")
            return

        logger.info(f"Saving FAISS index to disk at {self.index_path}")
        self.vectorstore.save_local(str(self.index_path))

    def similarity_search(self, query: str, k: int = None):
        """
        Perform similarity search.
        """
        if k is None:
            k = settings.TOP_K

        if self.vectorstore is None:
            logger.warning("FAISS vectorstore not initialized. Returning empty results.")
            return []

        logger.info(f"Performing similarity search | top_k={k}")
        return self.vectorstore.similarity_search(query, k=k)