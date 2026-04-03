from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

import json
import time
from pathlib import Path
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.rag_service import RAGService
from app.evaluation.ragas_mapper import RAGASMapper

logger = get_logger()
settings = get_settings()


class SingleQueryRAGASEvaluator:
    """
    Enterprise service that evaluates ONE query using RAGAS.
    Uses same LLM + embeddings from config (.env driven).
    Safe for rate-limited APIs (Gemini).
    """

    def __init__(self):
        self.rag_service = RAGService()
        self.log_file = Path(settings.LOG_PATH)
        self.log_file.parent.mkdir(exist_ok=True)

        # -------------------------------
        # Create Gemini evaluator LLM from config
        # -------------------------------
        logger.info(f"Initializing RAGAS evaluator LLM: {settings.LLM_MODEL}")

        gemini_llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0
        )
        self.ragas_llm = LangchainLLMWrapper(gemini_llm)

        # -------------------------------
        # Create embeddings from config (fallback safe)
        # -------------------------------
        embedding_model = getattr(settings, "RAGAS_EMBEDDING_MODEL", "models/embedding-001")
        logger.info(f"Initializing RAGAS embeddings: {embedding_model}")

        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model
        )
        self.ragas_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

    def evaluate(self, question: str, ground_truth: str):
        logger.info("Running single-query RAGAS evaluation")
        logger.info(f"Question: {question}")

        #  Query RAG system
        answer, documents = self.rag_service.query(question)

        #  Convert → RAGAS format
        sample = RAGASMapper.to_ragas_format(
            question,
            answer,
            documents,
            ground_truth
        )

        dataset = Dataset.from_list([sample])

        #  Run RAGAS metrics using Gemini LLM + embeddings
        scores = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
        )

        # Convert RAGAS result → JSON serializable dict
        score_dict = scores.to_pandas().iloc[0].to_dict()

        result = {
            "question": question,
            "ground_truth": ground_truth,
            "answer": answer,
            "scores": score_dict,
        }

        # Append result to JSONL log for record-keeping and analysis
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        logger.info("Evaluation saved to JSONL")

        # Respect rate limits
        time.sleep(settings.RAGAS_SLEEP_SECONDS)

        return result