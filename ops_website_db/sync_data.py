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

# Fix imports when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ops_website_db.firecrawl_client import FirecrawlClient
from ops_website_db.website_db import WebsiteDB

# Add fallback scraper imports
from bs4 import BeautifulSoup
import markdown

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("website_sync")

class ContentChunker:
    """Utility class for chunking content into smaller pieces."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: The text to chunk
            
        Returns:
            List of text chunks
        """
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []
            
        chunks = []
        start = 0
        
        while start < len(text):
            # Determine end of current chunk
            end = min(start + self.chunk_size, len(text))
            
            # If not at the end of text, try to find a good breaking point
            if end < len(text):
                # Look for paragraph break, then sentence break, then word break
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2
                else:
                    sentence_break = max(
                        text.rfind('. ', start, end),
                        text.rfind('.\n', start, end),
                        text.rfind('! ', start, end),
                        text.rfind('?\n', start, end)
                    )
                    if sentence_break != -1 and sentence_break > start + self.chunk_size // 2:
                        end = sentence_break + 2
                    else:
                        # Fall back to word boundary
                        space = text.rfind(' ', start, end)
                        if space != -1 and space > start + self.chunk_size // 2:
                            end = space + 1
            
            # Extract current chunk
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Move start position for next chunk, accounting for overlap
            start = max(start, end - self.chunk_overlap)
        
        return chunks

class FallbackScraper:
    """Fallback scraper using BeautifulSoup when the Firecrawl API fails."""
    
    def __init__(self):
        """Initialize the fallback scraper."""
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        self.timeout = 30
    
    def scrape(self, url: str) -> Dict[str, Any]:
        """
        Scrape a URL using BeautifulSoup.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with content and metadata
        """
        logger.info(f"Using fallback scraper for URL: {url}")
        
        try:
            # Fetch the page
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = soup.title.string if soup.title else ""
            description = ""
            meta_desc = soup.find("meta", attrs={"name": "description"})
            if meta_desc and "content" in meta_desc.attrs:
                description = meta_desc["content"]
                
            og_image = ""
            meta_og_image = soup.find("meta", attrs={"property": "og:image"})
            if meta_og_image and "content" in meta_og_image.attrs:
                og_image = meta_og_image["content"]
                
            # Extract main content
            # Remove script, style, and other non-content elements
            for element in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
                
            # Try to find the main content container
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '#content', '.content', '.main']:
                content = soup.select(selector)
                if content:
                    main_content = content[0]
                    break
            
            # If no specific content container found, use body
            if not main_content:
                main_content = soup.body
            
            # Get text content
            if main_content:
                # Clean whitespace
                text_content = re.sub(r'\s+', ' ', main_content.get_text().strip())
                
                # Convert to markdown-like format
                html_content = str(main_content)
                md_content = html_content  # Simple version just uses text
                
                # Build result dictionary
                return {
                    "content": {
                        "markdown": text_content,
                        "html": html_content
                    },
                    "metadata": {
                        "title": title,
                        "description": description,
                        "og": {
                            "image": og_image
                        }
                    },
                    "url": url,
                    "extracted_with": "fallback_scraper"
                }
            else:
                logger.warning(f"Could not find main content for URL: {url}")
                return {}
                
        except Exception as e:
            logger.error(f"Fallback scraper error for {url}: {str(e)}")
            return {}
    
    def crawl(self, url: str, **kwargs) -> Dict[str, Any]:
        """
        Simple crawling function that just scrapes the given URL.
        No actual crawling functionality in the fallback.
        
        Args:
            url: Starting URL
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Dictionary with scraped page
        """
        logger.info(f"Fallback crawler does not support full crawling. Only scraping {url}")
        
        result = self.scrape(url)
        if result:
            return {
                "pages": [result],
                "url": url,
                "extracted_with": "fallback_scraper"
            }
        return {"pages": []}

class WebsiteCrawler:
    """Crawler for websites using FirecrawlClient and storing in WebsiteDB."""
    
    SITE_CONFIGS = {
        "main": {
            "url": "https://tokenmetrics.com",
            "type": "main",
            "exclude_paths": ["/app/", "/login/", "/register/", "/reset-password/"],
            "max_depth": 2,
            "limit": 50
        },
        "research": {
            "url": "https://research.tokenmetrics.com",
            "type": "research",
            "exclude_paths": ["/wp-admin/", "/wp-login.php", "/feed/"],
            "max_depth": 2,
            "limit": 50
        },
        "blog": {
            "url": "https://newsletter.tokenmetrics.com",
            "type": "blog",
            "exclude_paths": ["/wp-admin/", "/wp-login.php", "/feed/"],
            "max_depth": 2,
            "limit": 50
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
        self.fallback_scraper = FallbackScraper()
        self.db = WebsiteDB()
        self.chunker = ContentChunker()
        self.use_fallback = os.getenv("USE_FALLBACK_SCRAPER", "auto").lower()
        self.max_wait_time = 600  # Default 10 minutes
    

    
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
        
        logger.info(f"Crawling site: {site_key}")
        logger.info(f"Starting crawl of {url} ({site_type})")
        
        # Check if fallback is explicitly enabled
        use_fallback = self.use_fallback == "true"
        
        # Initiate the crawl
        try:
            # Use Firecrawl unless fallback is explicitly enabled
            result = {}
            if not use_fallback:
                try:
                    # Start the crawl job
                    logger.info(f"Starting Firecrawl crawl for {url}...")
                    result = self.firecrawl.crawl(
                        url=url,
                        exclude_paths=site_config.get("exclude_paths", []),
                        max_depth=site_config.get("max_depth", 3),
                        limit=site_config.get("limit", 50),
                        ignore_sitemap=True,
                        ignore_query_parameters=True,
                        allow_external_links=False,
                        allow_backward_links=False,
                        # Don't wait for completion in initial call
                        wait_for_results=False
                    )
                    
                    # Handle asynchronous result (job ID returned)
                    if "success" in result and result.get("success") and "id" in result:
                        job_id = result.get("id")
                        job_url = result.get("url", "")
                        
                        logger.info(f"Crawl job started with ID: {job_id}")
                        logger.info(f"You can check status manually at: {job_url}")
                        
                        # Wait for completion
                        wait_time = self.max_wait_time 
                        logger.info(f"Waiting for crawl to complete (max {wait_time} seconds)...")
                        
                        # Poll for results
                        result = self.firecrawl.get_crawl_results(
                            job_id, 
                            wait_complete=True,
                            poll_interval=10,
                            max_wait_time=wait_time
                        )
                        
                        # Check if crawl completed successfully
                        if "status" in result and result["status"] != "completed":
                            logger.warning(f"Crawl did not complete within wait time. Status: {result.get('status')}")
                            if self.use_fallback == "auto":
                                logger.info("Using fallback scraper since crawl didn't complete")
                                use_fallback = True
                            else:
                                logger.info("Returning partial results without using fallback")
                        else:
                            logger.info(f"Crawl job completed with status: {result.get('status')}")
                
                except Exception as e:
                    logger.error(f"Firecrawl API error: {str(e)}")
                    if self.use_fallback == "auto":
                        logger.info("Using fallback scraper due to API error")
                        use_fallback = True
                    else:
                        logger.error("Fallback not enabled. Returning error.")
                        return {"error": str(e)}
            
            # Use fallback if needed and allowed
            if use_fallback:
                logger.info(f"Using fallback scraper for {url}")
                result = self.fallback_scraper.crawl(url=url)
            
            # Check if we got any pages
            if not result or "pages" not in result or not result["pages"]:
                error_message = "No pages returned from crawl"
                if "error" in result:
                    error_message += f": {result['error']}"
                if "status" in result and result["status"] != "completed":
                    error_message += f" (status: {result['status']})"
                    
                logger.error(f"Crawl failed for {url}: {error_message}")
                return {"error": error_message}
                
            pages = result["pages"]
            logger.info(f"Found {len(pages)} pages")
            
            # Process and store pages
            stored_count = 0
            chunk_count = 0
            
            # Process pages in smaller batches to avoid memory issues
            batch_size = 10  # Process 10 pages at a time
            total_pages = len(pages)
            
            for i in range(0, total_pages, batch_size):
                batch_pages = pages[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_pages+batch_size-1)//batch_size}: {len(batch_pages)} pages")
                
                for page in tqdm(batch_pages, desc=f"Processing {site_key} pages (batch {i//batch_size + 1})"):
                    page_url = page.get("url", "")
                    
                    # Skip if URL is invalid
                    if not page_url or not page_url.startswith(("http://", "https://")):
                        continue
                    
                    # Check if URL already exists and was recently updated
                    if not force_update:
                        url_record = self.db.get_url_record(page_url)
                        if url_record:
                            last_updated = url_record.get("last_updated")
                            if last_updated:
                                # Skip if updated in the last 24 hours
                                if datetime.now(last_updated.tzinfo) - last_updated < timedelta(hours=24):
                                    logger.debug(f"Skipping recently updated URL: {page_url}")
                                    continue
                    
                    # Extract content and metadata
                    content = page.get("content", {}).get("markdown", "")
                    title = page.get("metadata", {}).get("title", "")
                    
                    if not content:
                        logger.warning(f"No content for URL: {page_url}")
                        continue
                    
                    # Create chunks
                    chunks = self.chunker.chunk_text(content)
                    if not chunks:
                        logger.warning(f"No chunks created for URL: {page_url}")
                        continue
                    
                    # Extract metadata
                    metadata = self._extract_metadata(page)
                    
                    # Store in database
                    success = self.db.store_page_chunks(
                        website_type=site_type,
                        url=page_url,
                        title=title,
                        full_content=content,
                        chunks=chunks,
                        metadata=metadata
                    )
                    
                    if success:
                        stored_count += 1
                        chunk_count += len(chunks)
                
                # Clear memory after each batch
                batch_pages = None
                import gc
                gc.collect()
            
            return {
                "stored_urls": stored_count,
                "total_chunks": chunk_count,
                "total_pages": len(pages)
            }
                
        except Exception as e:
            logger.error(f"Error crawling site {site_key}: {str(e)}")
            logger.error(traceback.format_exc())
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
            
            # Check if fallback is explicitly enabled
            use_fallback = self.use_fallback == "true"
            
            # Try Firecrawl first if not explicitly using fallback
            result = {}
            if not use_fallback:
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
                    logger.info("Switching to fallback scraper")
                    use_fallback = True
            
            # Use fallback if needed
            if use_fallback or not result or "content" not in result:
                logger.info(f"Using fallback scraper for {url}")
                result = self.fallback_scraper.scrape(url=url)
            
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

    def extract_urls_from_sitemap(self, sitemap_url: str, output_file: str, filter_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract URLs from a sitemap.xml file and optionally filter by path
        
        Args:
            sitemap_url: URL of the sitemap
            output_file: File to save the extracted URLs
            filter_path: Optional path filter to include only URLs containing this path
            
        Returns:
            Dict with statistics about extracted URLs
        """
        logger.info(f"Extracting URLs from sitemap: {sitemap_url}")
        
        try:
            # Fetch the sitemap
            response = requests.get(sitemap_url, timeout=30)
            response.raise_for_status()
            
            # Parse the XML
            root = ET.fromstring(response.content)
            
            # Find namespace
            ns = {'sm': root.tag.split('}')[0].strip('{')} if '}' in root.tag else {}
            
            # Extract URLs
            urls = []
            if ns:
                url_elements = root.findall('.//sm:url/sm:loc', ns)
            else:
                url_elements = root.findall('.//url/loc')
                
            for url_elem in url_elements:
                url = url_elem.text.strip()
                
                # Apply filter if specified
                if filter_path is None or filter_path in url:
                    urls.append(url)
            
            # Save to file
            with open(output_file, 'w') as f:
                for url in urls:
                    f.write(f"{url}\n")
                    
            logger.info(f"Extracted {len(urls)} URLs (from {len(url_elements)} total)")
            logger.info(f"Saved URLs to {output_file}")
            
            return {
                "sitemap_url": sitemap_url,
                "total_urls": len(url_elements),
                "extracted_urls": len(urls),
                "output_file": output_file
            }
            
        except Exception as e:
            logger.error(f"Error extracting URLs from sitemap: {str(e)}")
            return {
                "sitemap_url": sitemap_url,
                "error": str(e)
            }

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
    crawl_parser.add_argument("--fallback", choices=["auto", "true", "false"],
                             default="auto", help="Whether to use fallback scraper (auto=only on failure)")
    crawl_parser.add_argument("--max-wait", type=int, 
                             help="Maximum wait time in seconds for crawl jobs (default: 600)")
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Scrape a single URL")
    scrape_parser.add_argument("url", help="URL to scrape")
    
    # Batch scrape command
    batch_scrape_parser = subparsers.add_parser("batch_scrape", help="Scrape multiple URLs from a file")
    batch_scrape_parser.add_argument("file", help="File with one URL per line")
    batch_scrape_parser.add_argument("--delay", type=int, default=2, 
                                    help="Delay in seconds between requests")
    
    # Sitemap command
    sitemap_parser = subparsers.add_parser("sitemap", help="Extract URLs from a sitemap")
    sitemap_parser.add_argument("url", help="URL of the sitemap.xml file")
    sitemap_parser.add_argument("--output", default="urls.txt", help="Output file for extracted URLs")
    sitemap_parser.add_argument("--filter", help="Only include URLs containing this path")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check database status")
    
    args = parser.parse_args()
    
    # Set wait time and fallback mode before initializing crawler
    os.environ["USE_FALLBACK_SCRAPER"] = args.fallback
    
    # Initialize crawler
    try:
        crawler = WebsiteCrawler()
        
        # Set max wait time from args
        if args.max_wait:
            crawler.max_wait_time = args.max_wait
        
        if args.command == "crawl":
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
                    
        elif args.command == "scrape":
            success = crawler.scrape_single_url(args.url)
            if success:
                print(f"Successfully scraped and stored URL: {args.url}")
            else:
                print(f"Failed to scrape URL: {args.url}")
                
        elif args.command == "batch_scrape":
            # Read URLs from file
            try:
                with open(args.file, 'r') as f:
                    urls = [line.strip() for line in f if line.strip()]
                    
                if not urls:
                    print(f"No URLs found in file: {args.file}")
                    sys.exit(1)
                
                print(f"Found {len(urls)} URLs to scrape")
                successful = 0
                failed = 0
                
                # Process each URL
                for i, url in enumerate(urls):
                    print(f"[{i+1}/{len(urls)}] Scraping: {url}")
                    try:
                        success = crawler.scrape_single_url(url)
                        if success:
                            successful += 1
                            print(f"✓ Successfully scraped: {url}")
                        else:
                            failed += 1
                            print(f"✗ Failed to scrape: {url}")
                        
                        # Sleep to avoid rate limiting
                        if i < len(urls) - 1:  # Don't sleep after the last URL
                            print(f"Waiting {args.delay} seconds before next request...")
                            time.sleep(args.delay)
                    except Exception as e:
                        failed += 1
                        print(f"✗ Error scraping {url}: {str(e)}")
                
                # Print summary
                print(f"\nBatch scrape completed: {successful} successful, {failed} failed")
                
            except Exception as e:
                print(f"Error reading URL file: {str(e)}")
                sys.exit(1)
                
        elif args.command == "sitemap":
            result = crawler.extract_urls_from_sitemap(
                sitemap_url=args.url,
                output_file=args.output,
                filter_path=args.filter
            )
            
            if "error" in result:
                print(f"Error extracting URLs: {result['error']}")
            else:
                print(f"Found {result['total_urls']} URLs in sitemap")
                print(f"Extracted {result['extracted_urls']} URLs" + 
                     (f" matching filter '{args.filter}'" if args.filter else ""))
                print(f"Saved to: {result['output_file']}")
                print(f"\nYou can now use these URLs with the batch_scrape command:")
                print(f"python -m ops_website_db.sync_data batch_scrape {result['output_file']}")
                
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
