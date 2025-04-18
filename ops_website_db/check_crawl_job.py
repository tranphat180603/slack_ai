#!/usr/bin/env python3
"""
Script to check the status and results of a Firecrawl crawl job by ID.
"""

import os
import sys
import json
import argparse
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import re
import markdown

from ops_website_db.website_db import WebsiteDB
from ops_website_db.sync_data import ContentChunker
from ops_website_db.firecrawl_client import FirecrawlClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("check_crawl_job")

import re


def clean_text(md: str) -> str:
    # 1. Convert Markdown to HTML
    html = markdown.markdown(md, extensions=['extra', 'smarty'])
    # 2. Strip HTML as above
    soup = BeautifulSoup(html, 'html.parser')
    return ' '.join(soup.get_text(separator=' ').split())


def check_crawl_job(job_id, save_results=False, output_file=None):
    """
    Check the status and results of a crawl job by ID.
    
    Args:
        job_id: The crawl job ID
        save_results: Whether to save the results to a file
        output_file: The file to save results to (if None, will use job_id.json)
        
    Returns:
        The job results (dict)
    """
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        logger.error("FIRECRAWL_API_KEY not found in environment")
        return None
        
    # Initialize client
    client = FirecrawlClient(api_key)
    
    logger.info(f"Checking crawl job: {job_id}")
    
    # Get job results
    result = client.get_crawl_results(job_id, wait_complete=False)
    
    # Print job information
    if "status" in result:
        status = result["status"]
        logger.info(f"Job status: {status}")
    else:
        logger.warning("No status information found")
    
    # Check for full crawl information
    if "_raw" in result:
        raw = result["_raw"]
        logger.info(f"Response keys: {', '.join(raw.keys())}")
        
        # Print summary stats
        if "total" in raw:
            logger.info(f"Total pages: {raw.get('total', 0)}")
        if "completed" in raw:
            logger.info(f"Completed pages: {raw.get('completed', 0)}")
        if "creditsUsed" in raw:
            logger.info(f"Credits used: {raw.get('creditsUsed', 0)}")
    
    # Check for pages
    if "pages" in result:
        pages = result["pages"]
        logger.info(f"Found {len(pages)} pages in the result")
        
        # Print sample of page data
        if pages:
            sample_content = pages[0].get("content", {}).get("markdown", "No content")
            for i, page in enumerate(pages):
                url = page.get("url", "No URL")
                title = page.get("metadata", {}).get("title", "No title")
                content = clean_text(page.get("content", {}).get("markdown", "No content"))
                logger.info(f"  {i+1}. {title} - {url}")
    else:
        logger.warning("No pages found in result")
    
    print(f"Sample page length: {len(sample_content)}")
    # website_db = WebsiteDB()
    content_chunker = ContentChunker()
    print("Chunking content...")
    for chunk in content_chunker.chunk_text(clean_text(sample_content)):
        print(chunk, end="\n\n", flush=True)
    
    return result

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Check the status and results of a Firecrawl crawl job")
    parser.add_argument("job_id", help="The crawl job ID to check")
    parser.add_argument("--save", action="store_true", help="Save the results to a file")
    parser.add_argument("--output", help="Output file name (default: job_id.json)")
    
    args = parser.parse_args()
    
    check_crawl_job(args.job_id, args.save, args.output)

if __name__ == "__main__":
    main() 