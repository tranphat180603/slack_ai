"""
Pipeline for crawling and syncing website content to the database.
Uses FirecrawlClient to crawl websites and WebsiteDB to store content.
"""
import os
import re
import sys
import time
import logging
import argparse
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tqdm import tqdm
import requests
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
import traceback
from typing import Iterator
import markdown
from bs4 import BeautifulSoup

# Fix imports when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ops_website_db.firecrawl_client import FirecrawlClient
from ops_website_db.website_db import WebsiteDB

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("website_sync")

class ContentChunker:
    """Utility class for chunking content into smaller pieces."""
    
    def __init__(self, chunk_size: int = 2000):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
        """
        self.chunk_size = chunk_size
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 2000,
    ) -> Iterator[str]:
        """
        Split text into overlapping chunks as a generator.
        
        Args:
            text: The text to chunk
            chunk_size: maximum size of each chunk
            
        Yields:
            One text chunk at a time.
        """
        if not text:
            return
        text_length = len(text)

        # If the whole text is shorter than a chunk, emit it once
        if text_length <= chunk_size:
            yield text
            return

        start = 0
        while start < text_length:
            end = min(start + chunk_size, text_length)

            # Try to find a good breakpoint if not at very end
            if end < text_length:
                search_start = start + chunk_size // 2

                # Paragraph break
                pb = text.find('\n\n', search_start, end)
                if pb != -1:
                    end = pb + 2
                else:
                    # Sentence break
                    for sep in ('. ', '.\n', '! ', '? '):
                        sb = text.rfind(sep, search_start, end)
                        if sb != -1:
                            end = sb + len(sep)
                            break
                    else:
                        # Word break
                        space = text.rfind(' ', search_start, end)
                        if space != -1:
                            end = space + 1

            # Yield the chunk
            chunk = text[start:end].strip()
            if chunk:
                yield chunk

            # Advance, keeping overlap
            start = max(end, start + 1)


class WebsiteCrawler:
    """Crawler for websites using FirecrawlClient and storing in WebsiteDB."""
    
    SITE_CONFIGS = {
        "main": {
            "url": "https://tokenmetrics.com",
            "type": "main",
            "exclude_paths": ["/app/", "/login/", "/register/", "/reset-password/"],
            "max_depth": 2,
            "limit": 20
        },
        "research": {
            "url": "https://research.tokenmetrics.com",
            "type": "research",
            "exclude_paths": ["/wp-admin/", "/wp-login.php", "/feed/"],
            "max_depth": 2,
            "limit": 20
        },
        "blog": {
            "url": "https://newsletter.tokenmetrics.com",
            "type": "blog",
            "exclude_paths": ["/wp-admin/", "/wp-login.php", "/feed/"],
            "max_depth": 2,
            "limit": 20
        }
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize the crawler.
        
        Args:
            api_key: Firecrawl API key (if None, will use FIRECRAWL_API_KEY env var)
        """
        if not api_key:
            api_key = os.getenv("FIRECRAWL_API_KEY")
            if not api_key:
                raise ValueError("FIRECRAWL_API_KEY not provided")
        
        self.firecrawl = FirecrawlClient(api_key)
        self.db = WebsiteDB()
        self.chunker = ContentChunker()
        self.max_wait_time = 600  # Default 10 minutes
    
    def clean_text(self, text: str) -> str:
        # 1. Convert Markdown to HTML
        html = markdown.markdown(text, extensions=['extra', 'smarty'])
        # 2. Strip HTML as above
        soup = BeautifulSoup(html, 'html.parser')
        return ' '.join(soup.get_text(separator=' ').split())
    
    def _get_site_type(self, url: str) -> str:
        """
        Determine site type from URL.
        
        Args:
            url: URL to check
            
        Returns:
            Site type (main, research, blog)
        """
        if "research.tokenmetrics.com" in url:
            return "research"
        elif "newsletter.tokenmetrics.com" in url:
            return "blog"
        elif "tokenmetrics.com" in url:
            return "main"
        else:
            return "unknown"
    
    def _extract_metadata(self, page_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from page data.
        
        Args:
            page_data: Page data from Firecrawl
            
        Returns:
            Extracted metadata
        """
        metadata = {}
        
        # Extract metadata from the page
        if "metadata" in page_data:
            # Common metadata
            if "title" in page_data["metadata"]:
                metadata["page_title"] = page_data["metadata"]["title"]
            if "description" in page_data["metadata"]:
                metadata["description"] = page_data["metadata"]["description"]
            if "author" in page_data["metadata"]:
                metadata["author"] = page_data["metadata"]["author"]
                
            # Handling OpenGraph metadata
            if "og" in page_data["metadata"]:
                og = page_data["metadata"]["og"]
                if "type" in og:
                    metadata["og_type"] = og["type"]
                if "image" in og:
                    metadata["og_image"] = og["image"]
        
        # Add crawl timestamp
        metadata["crawled_at"] = datetime.now().isoformat()
        
        return metadata
    
    def crawl_site(self, site_key: str, force_update: bool = False) -> Dict[str, Any]:
        """
        Crawl a specific site and store content in the database.
        
        Args:
            site_key: Site key (main, research, blog)
            force_update: Whether to force update all pages
            
        Returns:
            Dictionary with crawl statistics
        """
        if site_key not in self.SITE_CONFIGS:
            raise ValueError(f"Unknown site key: {site_key}")
        
        site_config = self.SITE_CONFIGS[site_key]
        url = site_config["url"]
        site_type = site_config["type"]
        logger.info(f"Starting crawl of {url} ({site_type})")

        try:
            # Kick off the crawl job
            result = self.firecrawl.crawl(
                url=url,
                exclude_paths=site_config.get("exclude_paths", []),
                max_depth=site_config.get("max_depth", 2),
                limit=site_config.get("limit", 50),
                ignore_sitemap=True,
                ignore_query_parameters=True,
                allow_external_links=False,
                allow_backward_links=False,
                wait_for_results=False
            )

            # Handle async job
            if not (result.get("success") and result.get("id")):
                return {"error": "Crawl could not be started."}
            job_id = result["id"]
            logger.info(f"Crawl job started with ID: {job_id}")

            # Wait for completion
            job_result = self.firecrawl.get_crawl_results(
                job_id,
                wait_complete=True,
                poll_interval=10,
                max_wait_time=self.max_wait_time
            )
            status = job_result.get("status")
            if status != "completed":
                logger.warning(f"Crawl did not complete: {status}")
                return {"error": f"Crawl status: {status}"}

            pages = job_result.get("pages", [])
            if not pages:
                return {"error": "No pages returned from crawl"}

            total_pages = len(pages)
            logger.info(f"Found {total_pages} pages")

            stored_count = 0
            chunk_count = 0
            
            # Process in batches to limit memory usage
            for i in range(0, total_pages):
                page = pages[i]
                page_url = page.get("url", "")
                if not page_url.startswith(("http://", "https://")):
                    continue

                # Skip recently updated
                if not force_update:
                    rec = self.db.get_url_record(page_url)
                    if rec and rec.get("last_updated"):
                        if datetime.now(rec["last_updated"].tzinfo) - rec["last_updated"] < timedelta(hours=24):
                            continue

                raw_md = page.get("content", {}).get("markdown", "")
                if not raw_md:
                    continue

                # Clean and split into chunks
                cleaned = self.clean_text(raw_md)
                chunks = list(self.chunker.chunk_text(cleaned, self.chunker.chunk_size))
                if not chunks:
                    continue

                meta = self._extract_metadata(page)
                success = self.db.store_page_chunks(
                    website_type=site_type,
                    url=page_url,
                    title=page.get("metadata", {}).get("title", ""),
                    full_content=raw_md,
                    chunks=chunks,
                    metadata=meta
                )
                if success:
                    stored_count += 1
                    chunk_count += len(chunks)

            # Free batch resources
            batch = None
            import gc; gc.collect()

            # Done
            return {
                "stored_urls": stored_count,
                "total_chunks": chunk_count,
                "total_pages": total_pages
            }

        except Exception as e:
            logger.error(f"Error crawling site {site_key}: {e}")
            return {"error": str(e)}

    
    def crawl_all_sites(self, force_update: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Crawl all configured sites.
        
        Args:
            force_update: Whether to force update of all pages
            
        Returns:
            Statistics about each crawl
        """
        stats = {}
        
        for site_key in self.SITE_CONFIGS:
            try:
                logger.info(f"Crawling site: {site_key}")
                site_stats = self.crawl_site(site_key, force_update)
                stats[site_key] = site_stats
                
                # Add a delay between site crawls to avoid rate limiting
                logger.info(f"Waiting 10 seconds before crawling the next site...")
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Error crawling site {site_key}: {str(e)}")
                stats[site_key] = {"error": str(e)}
        
        return stats
    
    def scrape_single_url(self, url: str) -> bool:
        """
        Scrape a single URL and store it in the database.
        
        Args:
            url: URL to scrape
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine site type
            site_type = self._get_site_type(url)
            if site_type == "unknown":
                logger.error(f"Unknown site type for URL: {url}")
                return False
            
            result = {}
            try:
                # Scrape the URL
                result = self.firecrawl.scrape(
                    url=url,
                    formats=["markdown"],
                    only_main_content=True,
                    remove_base64_images=True,
                    block_ads=True,
                    timeout=60000  # 60 seconds timeout
                )
            except Exception as e:
                logger.error(f"Firecrawl API error: {str(e)}")
            
            if not result or "content" not in result:
                logger.error(f"Scrape failed for URL: {url}")
                return False
            
            # Extract content and metadata
            content = result.get("content", {}).get("markdown", "")
            title = result.get("metadata", {}).get("title", "")
            
            if not content:
                logger.warning(f"No content for URL: {url}")
                return False
            
            # Create chunks
            chunks = self.chunker.chunk_text(content)
            if not chunks:
                logger.warning(f"No chunks created for URL: {url}")
                return False
            
            # Extract metadata
            metadata = self._extract_metadata(result)
            
            # Store in database
            success = self.db.store_page_chunks(
                website_type=site_type,
                url=url,
                title=title,
                full_content=content,
                chunks=chunks,
                metadata=metadata
            )
            
            if success:
                logger.info(f"Successfully scraped and stored URL: {url}")
                return True
            else:
                logger.error(f"Failed to store URL in database: {url}")
                return False
                
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            return False

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Website content crawling and syncing tool")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Crawl command
    crawl_parser = subparsers.add_parser("crawl", help="Crawl websites")
    crawl_parser.add_argument("--site", choices=["main", "research", "blog", "all"], 
                             default="all", help="Site to crawl")
    crawl_parser.add_argument("--force", action="store_true", 
                             help="Force update all pages, ignoring recent updates")
    crawl_parser.add_argument("--max-wait", type=int, 
                             help="Maximum wait time in seconds for crawl jobs (default: 600)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check database status")
    
    args = parser.parse_args()
        
    # Initialize crawler
    try:
        crawler = WebsiteCrawler()
        
        
        if args.command == "crawl":
            # Set max wait time from args
            if args.max_wait:
                crawler.max_wait_time = args.max_wait
            if args.site == "all":
                stats = crawler.crawl_all_sites(force_update=args.force)
                
                # Print summary
                print("\nCrawl Summary:")
                for site, site_stats in stats.items():
                    if "error" in site_stats:
                        print(f"- {site}: ERROR - {site_stats['error']}")
                    else:
                        print(f"- {site}: Stored {site_stats['stored_urls']} pages with {site_stats['total_chunks']} chunks")
            else:
                stats = crawler.crawl_site(args.site, force_update=args.force)
                
                # Print summary
                print("\nCrawl Summary:")
                if "error" in stats:
                    print(f"ERROR: {stats['error']}")
                else:
                    print(f"Stored {stats['stored_urls']} pages with {stats['total_chunks']} chunks")
                    
                
        elif args.command == "status":
            from ops_website_db.website_db import check_database_status
            check_database_status()
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
