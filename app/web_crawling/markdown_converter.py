import os
import re
from datetime import datetime
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger()
settings = get_settings()


class MarkdownConverter:
    """
    Converts crawled website html_content to Markdown files for ingestion.

    Responsibilities:
    - Convert HTML/text html_content to Markdown
    - Remove navigation / footer / junk links
    - Deduplicate repeated blocks
    - Save one markdown per URL into DATA_DIR
    """

    def __init__(self):
        self.output_dir = Path(settings.DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Markdown output directory set to: {self.output_dir}")

    def _sanitize_filename(self, url: str) -> str:
        """Convert URL to safe file name."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            path = "home"

        filename = re.sub(r"[^a-zA-Z0-9_-]", "_", path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{filename}.md"
    
    # ---------------------------------------------------------
    # CLEANING FUNCTIONS
    # ---------------------------------------------------------

    def _remove_images(self, md: str) -> str:
        """Remove images/icons from markdown."""
        return re.sub(r"!\[.*?\]\(.*?\)", "", md)

    def _remove_skip_links(self, md: str) -> str:
        """Remove skip-to-content links."""
        return re.sub(r"\[ ?Skip to content ?\]\(.*?\)", "", md)

    def _remove_link_clusters(self, md: str) -> str:
        """
        Remove navigation/footer blocks made of many links.
        """
        lines = md.split("\n")
        cleaned = []

        for line in lines:
            link_count = len(re.findall(r"\[.*?\]\(.*?\)", line))
            word_count = len(line.split())

            # If many links and little text → likely menu/footer
            if link_count > 2 and word_count < 50:
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    def _deduplicate_lines(self, md: str) -> str:
        """
        Remove repeated lines (homepages repeat sections a LOT).
        """
        seen = set()
        result = []

        for line in md.split("\n"):
            key = line.strip().lower()

            if not key:
                result.append(line)
                continue

            if key in seen:
                continue

            seen.add(key)
            result.append(line)

        return "\n".join(result)

    def _clean_markdown(self, md: str) -> str:
        """Full cleaning pipeline."""
        md = self._remove_images(md)
        md = self._remove_skip_links(md)
        md = self._remove_link_clusters(md)
        md = self._deduplicate_lines(md)

        # remove excessive blank lines
        md = re.sub(r"\n\s*\n\s*\n+", "\n\n", md)

        return md.strip()
    
    # ---------------------------------------------------------
    # Convert single page
    # ---------------------------------------------------------

    def save_markdown(self, page) -> Path:
        """
        Converts Crawl4AI CrawlResult → cleaned markdown file.

        Returns:
            Path: Path to saved markdown file
        """
        url = getattr(page, "url", None)
        title = getattr(page, "title", url) or url
        markdown_content = getattr(page, "markdown", None)

        if not markdown_content or len(markdown_content.strip()) < 10:
            logger.warning(f"Empty/low-content page skipped: {url}")
            return None

        # CLEAN the markdown to remove navigation, footers, duplicates
        markdown_content = self._clean_markdown(markdown_content)

        domain = urlparse(url).netloc

        # Create metadata header (VERY IMPORTANT FOR RAG)
        metadata = (
            f"---\n"
            f"url: {url}\n"
            f"domain: {domain}\n"
            f"title: {title}\n"
            f"date: {datetime.now().isoformat()}\n"
            f"---\n\n"
        )
        final_content = metadata + markdown_content

        # Sanitize and save file
        file_name = self._sanitize_filename(url)
        file_path = self.output_dir / file_name

        # Save file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_content)

        logger.info(f"Saved markdown for URL: {url} → {file_path}")
        return file_path

    def convert_pages(self, pages: List) -> List[Path]:
        """
        Converts multiple crawled pages to Markdown files.

        Args:
            pages (List): List of CrawlResult objects from crawler

        Returns:
            List[Path]: List of saved markdown file paths
        """
        saved_files = []
        for page in pages:
            try:
                file_path = self.save_markdown(page)
                if file_path:
                    saved_files.append(file_path)
            except Exception as e:
                url = getattr(page, 'url', 'Unknown')
                logger.error(f"Error converting page: {url} → {e}")
                
        logger.info(f"Total markdown files saved: {len(saved_files)}")
        return saved_files