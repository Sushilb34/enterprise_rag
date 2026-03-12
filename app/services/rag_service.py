from typing import List

from langchain_core.documents import Document

from app.main import EnterpriseRAG
from app.schemas.query import Source
from app.core.logger import get_logger

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
    
    def ingest(self, data_path: str):
        """
        Run document ingestion pipeline.
        """

        logger.info(f"Starting ingestion from {data_path}")

        chunks = self.rag.ingest_documents(data_path)

        logger.info("Ingestion completed.")

        return len(chunks)

    def query(self, question: str):
        """
        Query the RAG system and return answer + sources.
        """

        logger.info(f"Processing query: {question}")

        answer, documents = self.rag.ask_question(question)

        sources = self._extract_sources(documents)

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