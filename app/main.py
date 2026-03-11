from typing import List

from langchain_core.documents import Document

from app.core.logger import get_logger
from app.ingestion.loader import PDFLoader
from app.ingestion.splitter import DocumentSplitter
from app.llm.llm_provider import LLMProvider
from app.retrieval.reranker import CrossEncoderReranker
from app.vectorstore.hybrid_store import HybridRetriever

logger = get_logger()


class EnterpriseRAG:
    """
    Enterprise RAG Orchestrator

    Responsibilities:
    - Handle full ingestion pipeline
    - Handle query pipeline
    - Integrate Loader, Splitter, Embeddings, VectorStore, HybridRetriever, Reranker, LLM
    """

    def __init__(self):
        logger.info("Initializing Enterprise RAG system...")
        self.loader = PDFLoader()
        self.splitter = DocumentSplitter()
        self.llm = LLMProvider()
        self.retriever = None
        self.reranker = CrossEncoderReranker()
        self.documents: List[Document] = []
        logger.info("Enterprise RAG system initialized.")

    def ingest_documents(self):
        """
        Load PDFs, split into chunks, store in vectorstore + BM25
        """
        logger.info("Starting ingestion process...")

        # 1. Load PDFs
        docs = self.loader.load_pdfs()
        if not docs:
            logger.warning("No PDFs found for ingestion.")
            return

        # 2. Split documents
        chunks = self.splitter.split(docs)
        self.documents = chunks

        # 3. Initialize hybrid retriever
        self.retriever = HybridRetriever(chunks)

        logger.info(f"Ingestion completed | total chunks={len(chunks)}")

    def ask_question(self, query: str) -> str:
        """
        Query pipeline:
        1. Hybrid retrieval
        2. Reranking
        3. LLM generation
        """
        if self.retriever is None:
            # retriever hasn't been set up yet.  Try to load from an existing
            # vector index on disk rather than re-running the entire ingestion
            # pipeline; this keeps --query lightweight and avoids duplicates.
            logger.info("Retriever not initialized, attempting to load existing index.")
            self.retriever = HybridRetriever(None)
            if self.retriever is None:
                raise RuntimeError("Failed to initialize retriever.")

        logger.info(f"Received query: {query}")

        # 1. Hybrid retrieval
        retrieved_docs = self.retriever.retrieve(query)

        # 2. Rerank
        reranked_docs = self.reranker.rerank(query, retrieved_docs)

        # 3. Generate answer
        answer = self.llm.generate_answer(query, reranked_docs)

        logger.info("Query processed successfully.")

        return answer , reranked_docs


# Optional standalone test
if __name__ == "__main__":
    rag = EnterpriseRAG()
    if not rag.is_index_ready():
        rag.ingest_documents()

    while True:
        q = input("Enter your question (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        answer = rag.ask_question(q)
        print("\nAnswer:\n", answer, "\n")