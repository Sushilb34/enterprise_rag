import argparse
import sys

from app.main import EnterpriseRAG
from app.core.logger import get_logger

logger = get_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise RAG System Runner"
    )

    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run ingestion pipeline"
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Ask a question to the RAG system"
    )

    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Rebuild index from scratch"
    )

    args = parser.parse_args()

    rag = EnterpriseRAG()

    if args.reindex:
        logger.info("Reindexing from scratch...")
        rag.ingest_documents()
        logger.info("Reindexing completed.")
        sys.exit(0)

    if args.ingest:
        logger.info("Running ingestion pipeline...")
        rag.ingest_documents()
        logger.info("Ingestion completed.")
        sys.exit(0)

    if args.query:
        logger.info("Running query pipeline...")
        rag.ingest_documents()  # Ensure retriever is initialized
        answer = rag.ask_question(args.query)
        print("\nAnswer:\n")
        print(answer)
        sys.exit(0)

    parser.print_help()


if __name__ == "__main__":
    main()