import pandas as pd
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from app.core.logger import get_logger
from app.core.config import get_settings
from app.services.rag_service import RAGService
from app.evaluation.ragas_mapper import RAGASMapper

logger = get_logger()
settings = get_settings()


class RAGASRunner:
    """
    Enterprise RAGAS Evaluation Orchestrator
    """

    def __init__(self):
        self.dataset_path = settings.EVAL_RAGAS_PATH
        self.rag_service = RAGService()

    def _load_dataset(self) -> pd.DataFrame:
        """
        Load evaluation dataset from XLSX
        """
        df = pd.read_excel(self.dataset_path)

        if "question" not in df.columns:
            raise ValueError("Dataset must contain 'question' column")

        logger.info(f"RAGAS dataset loaded | size={len(df)}")

        return df

    def _generate_records(self, df: pd.DataFrame) -> list:
        """
        Generate RAG outputs and map to RAGAS format
        """
        records = []

        logger.info("Starting RAG inference for evaluation...")

        for _, row in df.iterrows():
            query = row["question"]
            ground_truth = row.get("ground_truth", "")

            answer, docs = self.rag_service.query(query)

            mapped = RAGASMapper.to_ragas_format(
                query=query,
                answer=answer,
                documents=docs,
                ground_truth=ground_truth
            )

            records.append(mapped)

        logger.info("RAG inference completed")

        return records

    def run(self):
        """
        Execute full RAGAS evaluation pipeline
        """

        df = self._load_dataset()

        records = self._generate_records(df)

        dataset = Dataset.from_list(records)

        logger.info("Running RAGAS evaluation...")

        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )

        logger.info(f"RAGAS Evaluation Results | {results}")

        return results