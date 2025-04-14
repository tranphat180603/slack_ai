#!/usr/bin/env python3
"""
Script to check the status and results of a Firecrawl crawl job by ID.
"""

import os
import sys
import json
import argparse
import logging
from dotenv import load_dotenv

# Fix imports when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ops_website_db.firecrawl_client import FirecrawlClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("check_crawl_job")

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
            logger.info("Sample pages:")
            for i, page in enumerate(pages[:-5]):  # Show last 5 pages
                url = page.get("url", "No URL")
                title = page.get("metadata", {}).get("title", "No title")
                content = page.get("content", {}).get("markdown", "No content")
                logger.info(f"  {i+1}. {title} - {url}")
                logger.info(f"    Content: {content[:200]}...")  # Show first 200 characters of content
                
            # Show number of remaining pages
            if len(pages) > 3:
                logger.info(f"  ...and {len(pages) - 3} more")
    else:
        logger.warning("No pages found in result")
    
    # Save results to file if requested
    if save_results:
        if not output_file:
            output_file = f"{job_id}.json"
            
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
    
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