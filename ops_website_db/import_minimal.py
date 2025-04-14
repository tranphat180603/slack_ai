#!/usr/bin/env python3
"""
Minimal script to import crawl data with extremely low memory usage.
Uses streaming JSON parsing and processes one item at a time.
"""

import os
import sys
import json
import ijson  # Add this import
import argparse
import logging
import time
import gc
import psutil
from typing import Dict, Any, List, Iterator
from datetime import datetime, timedelta

# Fix imports when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ops_website_db.website_db import WebsiteDB
from ops_website_db.sync_data import ContentChunker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("import_minimal")

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def detect_site_type(url: str) -> str:
    """Determine site type from URL"""
    if "research.tokenmetrics.com" in url:
        return "research"
    elif "newsletter.tokenmetrics.com" in url:
        return "blog"
    elif "tokenmetrics.com" in url:
        return "main"
    return "unknown"

def process_single_page(db, chunker, page: Dict[str, Any], site_type: str, force_update: bool) -> bool:
    """Process a single page and store it in the database"""
    try:
        logger.info("Begin processing a single page")
        log_memory_usage()
        
        page_url = page.get("url", "")
        logger.info(f"Processing URL: {page_url}")
        
        # Skip if URL is invalid
        if not page_url or not page_url.startswith(("http://", "https://")):
            logger.warning(f"Invalid URL, skipping: {page_url}")
            return False
        
        # Check if URL already exists and was recently updated
        if not force_update:
            url_record = db.get_url_record(page_url)
            if url_record:
                last_updated = url_record.get("last_updated")
                if last_updated:
                    # Skip if updated in the last 24 hours
                    if datetime.now(last_updated.tzinfo) - last_updated < timedelta(hours=24):
                        logger.debug(f"Skipping recently updated URL: {page_url}")
                        return False
        
        # Extract content and metadata
        logger.info("Extracting content")
        content = page.get("content", {}).get("markdown", "")
        title = page.get("metadata", {}).get("title", "")
        
        if not content:
            logger.warning(f"No content for URL: {page_url}")
            return False
        
        # Create chunks
        logger.info(f"Creating chunks for content (size: {len(content)} bytes)")
        chunks = chunker.chunk_text(content)
        if not chunks:
            logger.warning(f"No chunks created for URL: {page_url}")
            return False
        
        logger.info(f"Created {len(chunks)} chunks")
        
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
        logger.info("Storing page in database")
        success = db.store_page_chunks(
            website_type=site_type,
            url=page_url,
            title=title,
            full_content=content,
            chunks=chunks,
            metadata=metadata
        )
        
        # Clear variables to free memory
        logger.info("Clearing memory after page processing")
        content = None
        chunks = None
        metadata = None
        gc.collect()
        log_memory_usage()
        
        return success
    except Exception as e:
        logger.error(f"Error processing page {page.get('url', 'unknown')}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def stream_pages(file_path: str) -> Iterator[Dict[str, Any]]:
    """Stream pages from a JSON file without loading everything into memory"""
    try:
        count = 0
        with open(file_path, 'rb') as f:
            # First, try to determine if we have a pages array structure
            logger.info("Starting streaming pages from file")
            
            # Check for pages array
            pages_items = ijson.items(f, 'pages.item')
            for page in pages_items:
                count += 1
                if count % 10 == 0:
                    logger.info(f"Streamed {count} pages so far")
                yield page
                
            if count == 0:
                # If we didn't find pages array, check for a direct array of pages
                logger.info("No pages array found, checking for direct array")
                f.seek(0)
                pages_items = ijson.items(f, 'item')
                for page in pages_items:
                    count += 1
                    if count % 10 == 0:
                        logger.info(f"Streamed {count} pages so far")
                    yield page
        
        logger.info(f"Finished streaming {count} pages from file")
    except Exception as e:
        logger.error(f"Error streaming pages from file: {str(e)}")
        import traceback
        traceback.print_exc()
        yield from []

def count_pages(file_path: str) -> int:
    """Count pages in file to give progress feedback"""
    try:
        # Quick check using file size
        file_size = os.path.getsize(file_path)
        if file_size > 50 * 1024 * 1024:  # If file is > 50MB
            logger.info(f"Large file detected ({file_size / 1024 / 1024:.2f} MB), skipping page counting")
            return 0  # Skip counting for large files
            
        # For smaller files, try a quick check
        count = 0
        with open(file_path, 'r') as f:
            content = f.read(10000)  # Read first 10KB
            # Rough estimate by counting "url" occurrences
            count = content.count('"url"')
            if count > 5:
                logger.info(f"Estimated around {count} pages (rough count)")
                return count
            
        return 0
    except Exception as e:
        logger.error(f"Error counting pages: {str(e)}")
        return 0

def import_file(file_path: str, force_update: bool = False, batch_size: int = 1) -> Dict[str, Any]:
    """
    Import crawl data from a JSON file into the database using streaming.
    
    Args:
        file_path: Path to the JSON file with crawl results
        force_update: Whether to force update existing pages
        batch_size: Number of pages to process in each batch
        
    Returns:
        Dictionary with import statistics
    """
    logger.info(f"Importing data from file: {file_path}")
    log_memory_usage()
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return {"error": f"File not found: {file_path}"}
    
    # Log file size
    file_size = os.path.getsize(file_path)
    logger.info(f"File size: {file_size / 1024 / 1024:.2f} MB")
    
    # Initialize database and chunker
    db = WebsiteDB()
    chunker = ContentChunker()
    
    # Initialize counters
    stored_count = 0
    skipped_count = 0
    site_type = "unknown"
    url_detected = False
    
    try:
        # Get rough page count for progress feedback
        estimated_pages = count_pages(file_path)
        
        # Stream and process pages
        for i, page in enumerate(stream_pages(file_path)):
            # Set site type from first page with URL
            if not url_detected and page.get("url"):
                site_type = detect_site_type(page.get("url"))
                logger.info(f"Site type detected: {site_type}")
                url_detected = True
            
            # Log progress
            if i % 5 == 0:
                if estimated_pages > 0:
                    logger.info(f"Processing page {i+1}/{estimated_pages}")
                else:
                    logger.info(f"Processing page {i+1}")
                log_memory_usage()
            
            success = process_single_page(db, chunker, page, site_type, force_update)
            
            if success:
                stored_count += 1
            else:
                skipped_count += 1
            
            # Clear memory after each page
            page = None
            gc.collect()
            
            # Brief pause to let the database catch up
            if i % batch_size == 0 and i > 0:
                time.sleep(0.5)
        
        logger.info(f"Import completed: {stored_count} pages stored, {skipped_count} pages skipped")
        log_memory_usage()
        
        return {
            "stored_urls": stored_count,
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
    parser = argparse.ArgumentParser(description="Minimal memory streaming import script for crawl data")
    
    # Basic arguments
    parser.add_argument("file", help="JSON file containing crawl results")
    parser.add_argument("--force", action="store_true", help="Force update all pages, ignoring recent updates")
    parser.add_argument("--batch-size", type=int, default=1, help="Pages to process before pausing (default: 1)")
    
    args = parser.parse_args()
    
    # Import file
    stats = import_file(args.file, args.force, args.batch_size)
    
    if "error" in stats:
        logger.error(f"Error importing file {args.file}: {stats['error']}")
        sys.exit(1)
    else:
        logger.info(f"File {args.file} import results: {stats}")
        sys.exit(0)

if __name__ == "__main__":
    main() 