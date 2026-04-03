import time
from typing import List

from langchain_core.documents import Document

from app.core.logger import get_logger
from app.evaluation.eval_logger import EvaluationLogger
from app.guardrails.answer_guard import AnswerGuardrail
from app.ingestion.loader import DocumentLoader
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
        self.loader = DocumentLoader()
        self.splitter = DocumentSplitter()
        self.llm = LLMProvider()
        self.retriever = None
        self.reranker = CrossEncoderReranker()
        self.documents: List[Document] = []
        self.answer_guardrail = AnswerGuardrail()
        self.eval_logger = EvaluationLogger()
        logger.info("Enterprise RAG system initialized.")

    def ingest_documents(self):
        """
        Load PDFs, split into chunks, store in vectorstore + BM25
        """
        logger.info("Starting ingestion process...")

        # 1. Load documents (PDFs and Markdown files)
        docs = self.loader.load()
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
        total_start = time.perf_counter()

        # 1. Hybrid retrieval
        retrieval_start = time.perf_counter()
        retrieved_docs = self.retriever.retrieve(query)
        retrieval_time = time.perf_counter() - retrieval_start

        # 2. Rerank
        rerank_start = time.perf_counter()
        reranked_docs = self.reranker.rerank(query, retrieved_docs)
        rerank_time = time.perf_counter() - rerank_start

        # 3. Generate answer
        llm_start = time.perf_counter()
        answer = self.llm.generate_answer(query, reranked_docs)
        answer, guardrail_triggered = self.answer_guardrail.apply(
            answer, reranked_docs
        )

        llm_time = time.perf_counter() - llm_start

        total_time = time.perf_counter() - total_start
    
        logger.info(
            f"PIPELINE METRICS | "
            f"retrieval={retrieval_time:.3f}s | "
            f"rerank={rerank_time:.3f}s | "
            f"llm={llm_time:.3f}s | "
            f"total={total_time:.3f}s | "
            f"retrieved_docs={len(retrieved_docs)} | "
            f"reranked_docs={len(reranked_docs)}"
        )

        eval_data = {
            "query": query,
            "retrieved_docs": len(retrieved_docs),
            "top_hybrid_score": retrieved_docs[0].metadata.get("hybrid_score", 0.0) if retrieved_docs else 0.0,
            "reranked_docs": len(reranked_docs),
            "top_rerank_score": reranked_docs[0].metadata.get("rerank_score", 0.0) if reranked_docs else 0.0,
            "answer_length": len(answer.split()),
            "guardrail_triggered": guardrail_triggered,
            "latency": round(total_time, 3)
        }
        self.eval_logger.log(eval_data)
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