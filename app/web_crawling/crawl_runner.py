import asyncio

from app.core.logger import get_logger
from app.web_crawling.crawler_service import WebsiteCrawlerService

logger = get_logger()


async def run_crawl_pipeline():
    """
    Runs the FULL async crawl pipeline.

    WebsiteCrawlerService now handles:
    - URL discovery
    - Crawling
    - Markdown conversion
    """

    logger.info("====================================")
    logger.info("Starting Website Crawling Pipeline (ASYNC)")
    logger.info("====================================")

    crawler = WebsiteCrawlerService()

    saved_files = await crawler.crawl_and_convert()

    if not saved_files:
        logger.warning("No files were created. Pipeline finished with no output.")
        return []

    logger.info("====================================")
    logger.info("Crawl Pipeline Completed Successfully")
    logger.info(f"Markdown files created: {len(saved_files)}")
    logger.info("Files are ready for ingestion by RAG pipeline")
    logger.info("====================================")

    return saved_files


def main():
    saved_files = asyncio.run(run_crawl_pipeline())

    print("\n Crawl complete!")
    print(f"Markdown files created: {len(saved_files)}")

    if saved_files:
        print(f"First file saved at: {saved_files[0]}")


if __name__ == "__main__":
    main()