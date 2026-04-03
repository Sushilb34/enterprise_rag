import asyncio
import xml.etree.ElementTree as ET
from typing import List, Set
from urllib.parse import urljoin, urlparse

from crawl4ai import (AsyncUrlSeeder, AsyncWebCrawler, CrawlerRunConfig,
                      SeedingConfig)
from crawl4ai.async_configs import BrowserConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from app.core.config import get_settings
from app.core.logger import get_logger
from app.web_crawling.markdown_converter import MarkdownConverter

logger = get_logger()
settings = get_settings()


class WebsiteCrawlerService:
    """
    Enterprise website crawler powered by Crawl4AI.

    Responsibilities:
    - Discover URLs via sitemap
    - Restrict crawling to allowed domains
    - Crawl pages asynchronously
    - Send pages to MarkdownConverter
    """

    def __init__(self):
        self.base_url = settings.CRAWL_BASE_URL.rstrip("/")
        self.max_pages = settings.CRAWL_MAX_PAGES
        self.max_depth = settings.CRAWL_MAX_DEPTH
        self.converter = MarkdownConverter()

        self.browser_config = BrowserConfig(
            headless=True,
            verbose=False
        )

        self.run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            exclude_external_images=True,
            exclude_social_media_links=True,
            excluded_tags=["nav", "footer", "aside", "header", "form", "input", "button", "svg", "canvas",],
        )

        self.extraction_strategy = LLMExtractionStrategy(
            # remove headers / nav / footer / junk automatically
            content_filter=PruningContentFilter(
                threshold=0.45,
                threshold_type="dynamic",
                min_word_threshold=50
            ),

            # convert HTML → Markdown automatically
            markdown_generator=DefaultMarkdownGenerator(),
        )
        

    # -----------------------------------------------------
    # Sitemap discovery
    # -----------------------------------------------------
    def _get_sitemap_url(self) -> str:
        """Guess sitemap.xml location."""
        return urljoin(self.base_url, "/wp-sitemap.xml")

    async def fetch_sitemap_urls(
        self,
        max_urls: int = 200
    ) -> Set[str]:
        """Fetch and parse sitemap(s) using crawl4ai's AsyncUrlSeeder."""
        all_urls = set()
        
        try:
            logger.info(f"Using AsyncUrlSeeder for {self.base_url}")
            async with AsyncUrlSeeder() as seeder:
                config = SeedingConfig(
                    source="sitemap+cc",  # Use sitemap and Common Crawl for maximum coverage
                    max_urls=max_urls
                )
                
                # Extract domain for seeder
                domain = urlparse(self.base_url).netloc
                if not domain:
                    domain = self.base_url
                
                discovered_urls = await seeder.urls(domain, config)
                
                for item in discovered_urls:
                    if isinstance(item, dict) and "url" in item:
                        all_urls.add(item["url"])
                    elif isinstance(item, str):
                        all_urls.add(item)
                        
            logger.info(f"AsyncUrlSeeder discovered {len(all_urls)} URLs")
        except Exception as e:
            logger.error(f"AsyncUrlSeeder failed: {str(e)}")
            # Fallback to a very basic sitemap.xml fetch if seeder fails
            try:
                sitemap_url = urljoin(self.base_url, "/wp-sitemap.xml")
                async with AsyncWebCrawler() as crawler:
                    result = await crawler.arun(url=sitemap_url)
                    if result.success and result.html:
                        urls = await self._parse_sitemap_xml(result.html)
                        all_urls.update(urls)
            except Exception as se:
                logger.error(f"Manual sitemap fallback failed: {str(se)}")
        logger.info(f"Total URLs discovered from sitemap: {len(all_urls)}")
        return all_urls

    async def _parse_sitemap_xml(self, xml_content: str) -> Set[str]:
        """Parse XML sitemap and extract all URLs. Handles multiple formats and malformed XML."""
        urls = set()
        if not xml_content:
            return urls

        # 1. Primary extraction: Use regex (very robust for sitemaps/feeds)
        import re

        # Extract from <loc> tags (Sitemaps)
        loc_urls = re.findall(r'<loc>\s*(https?://[^<>\s"\' ]+)\s*</loc>', xml_content, re.IGNORECASE)
        urls.update(u.strip() for u in loc_urls if u.strip())
        
        # Extract from <link> tags (RSS/Atom)
        # Handle <link>URL</link>
        link_tag_urls = re.findall(r'<link>\s*(https?://[^<>\s"\' ]+)\s*</link>', xml_content, re.IGNORECASE)
        urls.update(u.strip() for u in link_tag_urls if u.strip())
        
        # Handle <link href="URL" ... />
        link_href_urls = re.findall(r'<link[^>]+href=["\'](https?://[^"\'\s>]+)["\']', xml_content, re.IGNORECASE)
        urls.update(u.strip() for u in link_href_urls if u.strip())

        # 2. Secondary extraction: XML parsing (more precise if it works)
        try:
            # Clean up content for XML parser
            # Strip ALL namespaces to make findall work without namespaces
            cleaned_xml = re.sub(r'\s+xmlns(:\w+)?=["\'][^"\']+["\']', '', xml_content)
            # Remove XML declaration if it causes issues
            cleaned_xml = re.sub(r'<\?xml[^?]+\?>', '', cleaned_xml).strip()
            
            if cleaned_xml:
                # Add a dummy root if it looks like a list of fragments
                if not cleaned_xml.startswith('<'):
                    cleaned_xml = f"<root>{cleaned_xml}</root>"
                
                try:
                    root = ET.fromstring(cleaned_xml)
                    
                    # Search for any 'loc' or 'link' tags regardless of depth
                    for elem in root.iter():
                        tag = elem.tag.split('}')[-1].lower() if '}' in elem.tag else elem.tag.lower()
                        if tag == 'loc':
                            if elem.text and elem.text.strip():
                                urls.add(elem.text.strip())
                        elif tag == 'link':
                            # check text
                            if elem.text and elem.text.strip():
                                urls.add(elem.text.strip())
                            # check href attribute
                            href = elem.get('href')
                            if href and href.strip():
                                urls.add(href.strip())
                except ET.ParseError:
                    # If XML parsing fails, we already have regex results
                    pass
                    
        except Exception:
            # Silence all secondary parsing errors since we have regex fallback
            pass
        
        if urls:
            logger.info(f"Discovered {len(urls)} URLs from sitemap/feed content")
        
        return urls

    # -----------------------------------------------------
    # Filter URLs by domain & deduplicate
    # -----------------------------------------------------
    def _filter_urls(self, urls: List[str]) -> List[str]:
        """Keep URLs under the same domain only and remove duplicates."""
        parsed_base = urlparse(self.base_url)
        base_domain = parsed_base.netloc

        filtered: Set[str] = set()
        for url in urls:
            if urlparse(url).netloc == base_domain:
                filtered.add(url.rstrip("/"))

        return list(filtered)

    # -----------------------------------------------------
    # Crawl all URLs
    # -----------------------------------------------------
    async def crawl_all_urls(self, urls: List[str]) -> List:
        """Crawl each URL concurrently and return all results."""
        all_results = []

        # Limit concurrent crawls to prevent browser overload
        semaphore = asyncio.Semaphore(5)  # Allow max 5 concurrent crawls
        
        async def crawl_with_semaphore(url):
            async with semaphore:
                async with AsyncWebCrawler(config=self.browser_config) as crawler:
                    try:
                        result = await crawler.arun(
                            url=url,
                            config=self.run_config,
                            extraction_strategy=self.extraction_strategy,
                            max_pages=1,
                        )
                        return result
                    except Exception as e:
                        logger.warning(f"Failed to crawl {url}: {e}")
                        return None

        tasks = [crawl_with_semaphore(url) for url in urls]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        valid_results = [r for r in all_results if r is not None and not isinstance(r, Exception)]
        
        logger.info(f"Total pages crawled: {len(valid_results)}")
        return valid_results

    # -----------------------------------------------------
    # Main crawl & conversion entrypoint
    # -----------------------------------------------------
    async def crawl_and_convert(self):
        """
        Full pipeline (fully async):
        1) Discover URLs from sitemap
        2) Filter by domain
        3) Crawl all pages asynchronously
        4) Convert each page to Markdown asynchronously
        """
        logger.info("Starting website crawl pipeline")

        # async with AsyncWebCrawler() as crawler:
            # Step 1 — Discover URLs
        urls = await self.fetch_sitemap_urls()
        urls = self._filter_urls(urls)
        logger.info(f"Filtered URLs: {len(urls)}")

        # Step 2 — Crawl all URLs asynchronously
        pages = await self.crawl_all_urls(urls)

        markdown_files = []

        logger.info("Converting crawled pages → Markdown")

        for page in pages:
            try:
                saved_files = self.converter.convert_pages([page])
                if saved_files:
                    markdown_files.extend(saved_files)
            except Exception as e:
                logger.warning(f"Markdown conversion failed for {page.url}: {e}")

        logger.info(f"Markdown files created: {len(markdown_files)}")
        return markdown_files


# -----------------------------------------------------
# Async runner helper
# -----------------------------------------------------
def run_crawl_pipeline():
    service = WebsiteCrawlerService()
    return asyncio.run(service.crawl_and_convert())


if __name__ == "__main__":
    run_crawl_pipeline()