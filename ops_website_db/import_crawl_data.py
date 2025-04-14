#!/usr/bin/env python3
"""
Script to import crawl data from saved JSON files into the database.
This handles the database insertion process separately from crawling.
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import List, Dict, Any
from tqdm import tqdm
from datetime import datetime, timedelta

# Fix imports when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ops_website_db.website_db import WebsiteDB
from ops_website_db.sync_data import ContentChunker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("import_crawl_data")

def import_from_job_id(job_id: str, force_update: bool = False, batch_size: int = 10) -> Dict[str, Any]:
    """
    Import crawl data from a saved job ID JSON file into the database.
    
    Args:
        job_id: The crawl job ID (file should be named <job_id>.json)
        force_update: Whether to force update existing pages
        batch_size: Number of pages to process in each batch
        
    Returns:
        Dictionary with import statistics
    """
    # Check if file exists in current directory
    file_path = f"{job_id}.json"
    if not os.path.exists(file_path):
        # Try in the ops_website_db directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, f"{job_id}.json")
        if not os.path.exists(file_path):
            logger.error(f"File not found: {job_id}.json in current directory or in {script_dir}")
            return {"error": f"File not found: {job_id}.json"}
    
    return import_from_file(file_path, force_update, batch_size)

def import_from_file(file_path: str, force_update: bool = False, batch_size: int = 10) -> Dict[str, Any]:
    """
    Import crawl data from a JSON file into the database.
    
    Args:
        file_path: Path to the JSON file with crawl results
        force_update: Whether to force update existing pages
        batch_size: Number of pages to process in each batch
        
    Returns:
        Dictionary with import statistics
    """
    logger.info(f"Importing data from file: {file_path}")
    
    try:
        # Load the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Initialize database and chunker
        db = WebsiteDB()
        chunker = ContentChunker()
        
        # Check if we have pages
        if not data or "pages" not in data or not data["pages"]:
            logger.error(f"No pages found in file: {file_path}")
            return {"error": "No pages found in file"}
        
        pages = data["pages"]
        logger.info(f"Found {len(pages)} pages to import")
        
        # Determine site type from the first page
        site_type = "unknown"
        if pages and pages[0].get("url"):
            first_url = pages[0]["url"]
            if "research.tokenmetrics.com" in first_url:
                site_type = "research"
            elif "newsletter.tokenmetrics.com" in first_url:
                site_type = "blog"
            elif "tokenmetrics.com" in first_url:
                site_type = "main"
        
        logger.info(f"Site type detected: {site_type}")
        
        # Process and store pages
        stored_count = 0
        chunk_count = 0
        skipped_count = 0
        
        # Process pages in batches
        total_pages = len(pages)
        for i in range(0, total_pages, batch_size):
            batch_pages = pages[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_pages+batch_size-1)//batch_size}: {len(batch_pages)} pages")
            
            for page in tqdm(batch_pages, desc=f"Processing pages (batch {i//batch_size + 1})"):
                page_url = page.get("url", "")
                
                # Skip if URL is invalid
                if not page_url or not page_url.startswith(("http://", "https://")):
                    logger.warning(f"Invalid URL, skipping: {page_url}")
                    skipped_count += 1
                    continue
                
                # Check if URL already exists and was recently updated
                if not force_update:
                    url_record = db.get_url_record(page_url)
                    if url_record:
                        last_updated = url_record.get("last_updated")
                        if last_updated:
                            # Skip if updated in the last 24 hours
                            if datetime.now(last_updated.tzinfo) - last_updated < timedelta(hours=24):
                                logger.debug(f"Skipping recently updated URL: {page_url}")
                                skipped_count += 1
                                continue
                
                # Extract content and metadata
                content = page.get("content", {}).get("markdown", "")
                title = page.get("metadata", {}).get("title", "")
                
                if not content:
                    logger.warning(f"No content for URL: {page_url}")
                    skipped_count += 1
                    continue
                
                # Create chunks
                chunks = chunker.chunk_text(content)
                if not chunks:
                    logger.warning(f"No chunks created for URL: {page_url}")
                    skipped_count += 1
                    continue
                
                # Extract metadata
                metadata = {}
                if "metadata" in page:
                    meta = page["metadata"]
                    metadata = {
                        "page_title": meta.get("title", ""),
                        "description": meta.get("description", ""),
                        "author": meta.get("author", ""),
                        "crawled_at": datetime.now().isoformat()
                    }
                    
                    # OpenGraph metadata if available
                    if "og" in meta:
                        og = meta["og"]
                        metadata["og_type"] = og.get("type", "")
                        metadata["og_image"] = og.get("image", "")
                
                # Store in database
                success = db.store_page_chunks(
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
                else:
                    logger.warning(f"Failed to store page: {page_url}")
                    skipped_count += 1
            
            # Clear memory after each batch
            batch_pages = None
            import gc
            gc.collect()
            
            # Short pause between batches to let the database catch up
            time.sleep(1)
        
        logger.info(f"Import completed: {stored_count} pages stored, {chunk_count} chunks created, {skipped_count} pages skipped")
        
        return {
            "stored_urls": stored_count,
            "total_chunks": chunk_count,
            "skipped_urls": skipped_count,
            "site_type": site_type
        }
            
    except Exception as e:
        logger.error(f"Error importing data from {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Import crawl data into the database")
    
    # Script can take either job IDs or file paths
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job-id", nargs='+', help="Crawl job ID(s) to import (must have .json file with same name)")
    group.add_argument("--file", nargs='+', help="JSON file path(s) containing crawl results")
    
    # Other options
    parser.add_argument("--force", action="store_true", help="Force update all pages, ignoring recent updates")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of pages to process in each batch")
    
    args = parser.parse_args()
    
    if args.job_id:
        logger.info(f"Importing data from {len(args.job_id)} job ID(s)")
        for job_id in args.job_id:
            logger.info(f"Processing job ID: {job_id}")
            stats = import_from_job_id(job_id, args.force, args.batch_size)
            
            if "error" in stats:
                logger.error(f"Error importing job {job_id}: {stats['error']}")
            else:
                logger.info(f"Job {job_id} import results: {stats}")
    
    elif args.file:
        logger.info(f"Importing data from {len(args.file)} file(s)")
        for file_path in args.file:
            logger.info(f"Processing file: {file_path}")
            stats = import_from_file(file_path, args.force, args.batch_size)
            
            if "error" in stats:
                logger.error(f"Error importing file {file_path}: {stats['error']}")
            else:
                logger.info(f"File {file_path} import results: {stats}")
    
if __name__ == "__main__":
    main() 