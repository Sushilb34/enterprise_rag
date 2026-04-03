import argparse
from app.core.logger import get_logger
from app.evaluation.single_ragas_evaluator import SingleQueryRAGASEvaluator

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Run single-query RAGAS evaluation"
    )

    parser.add_argument("--question", required=True)
    parser.add_argument("--ground_truth", required=True)

    args = parser.parse_args()

    evaluator = SingleQueryRAGASEvaluator()

    result = evaluator.evaluate(
        question=args.question,
        ground_truth=args.ground_truth
    )

    logger.info("Final Scores:")
    logger.info(result["scores"])


if __name__ == "__main__":
    main()