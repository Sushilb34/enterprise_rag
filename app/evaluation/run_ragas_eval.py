from app.evaluation.ragas_runner import RAGASRunner
from app.core.logger import get_logger

logger = get_logger()

def main():
    logger.info("Starting RAGAS evaluation pipeline...")

    runner = RAGASRunner()
    results = runner.run()

    logger.info("Evaluation completed successfully.")
    logger.info(f"Final RAGAS Scores: {results}")

if __name__ == "__main__":
    main()