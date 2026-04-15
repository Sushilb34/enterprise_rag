from typing import List
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import get_settings
from app.main import EnterpriseRAG
from app.schemas.query import Source
from app.core.logger import get_logger

settings = get_settings()
logger = get_logger()


class RAGService:
    """
    Service layer for interacting with the Enterprise RAG pipeline.

    Responsibilities:
    - Call the RAG pipeline
    - Format responses for the API
    - Extract document sources
    """

    def __init__(self):
        logger.info("Initializing RAG Service...")
        self.rag = EnterpriseRAG()
        logger.info("RAG Service initialized successfully.")
    
    def ingest(self, data_dir: str):
        """
        Run document ingestion pipeline.
        """

        logger.info(f"Starting ingestion from {data_dir}")

        chunks = self.rag.ingest_documents()

        logger.info("Ingestion completed.")

        return True
    
    def reindex(self):
        """
        Rebuild the entire index from scratch.

        This deletes existing FAISS and BM25 indexes,
        reinitializes the RAG system, and runs ingestion again.
        """

        logger.info("Starting full reindex process...")

        # Delete FAISS index files
        faiss_path = Path(settings.FAISS_INDEX_PATH)

        if faiss_path.exists():
            logger.info("Deleting existing FAISS index files...")
            for file in faiss_path.glob("*"):
                file.unlink()

        # Delete BM25 index
        bm25_path = Path(settings.BM25_INDEX_PATH)

        if bm25_path.exists():
            logger.info("Deleting existing BM25 index file...")
            bm25_path.unlink()

            metadata_path =  Path(str(bm25_path) + ".meta")
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info("BM25 metadata file deleted successfully.")
            else:
                logger.info("BM25 metadata file NOT found.")

        logger.info("Old indexes removed successfully.")

        # Reinitialize the RAG pipeline
        logger.info("Reinitializing Enterprise RAG pipeline...")
        self.rag = EnterpriseRAG()

        # Run ingestion again
        logger.info("Running ingestion to rebuild indexes...")
        chunks = self.rag.ingest_documents()

        logger.info(f"Reindex completed successfully")

        return True
    

    def query(self, question: str):
        """
        Query the RAG system and return answer + sources.
        """

        logger.info(f"Processing query: {question}")

        answer, documents = self.rag.ask_question(question)

        # Smart Source Filtering:
        # Only show sources if the top document has a high enough relevance score.
        # This prevents showing random sources for simple greetings.
        sources = []
        if documents:
            top_score = documents[0].metadata.get("rerank_score", -99.0)
            logger.info(f"Top rerank score: {top_score:.4f}")
            
            if top_score >= 0.0: # Threshold for ms-marco-MiniLM-L-12-v2
                sources = self._extract_sources(documents)
            else:
                logger.info("Relevance score too low. Hiding sources (likely small talk).")

        return answer, sources
    
    

    def _extract_sources(self, documents: List[Document]) -> List[Source]:
        """
        Extract citation sources from retrieved documents.
        """

        sources = []

        for doc in documents:
            file_name = doc.metadata.get("file_name")
            page_number = doc.metadata.get("page_number")

            sources.append(
                Source(
                    file_name=file_name,
                    page_number=page_number
                )
            )

        return sources