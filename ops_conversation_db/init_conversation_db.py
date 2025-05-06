#!/usr/bin/env python3
"""
Initialize conversation database tables
"""

from ops_conversation_db.conversation_db import init_db, check_db_connection
import logging
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def initialize_db(max_retries=5):
    """Initialize database tables with retry logic"""
    for attempt in range(1, max_retries + 1):
        logger.info(f"Attempt {attempt}/{max_retries} to initialize conversation database...")
        
        # Check if we can connect to the database
        if not check_db_connection():
            logger.error(f"Failed to connect to database on attempt {attempt}")
            if attempt < max_retries:
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)
                continue
            else:
                logger.error("Max retries reached, giving up")
                return False
        
        # Initialize the database tables
        try:
            logger.info("Creating conversation database tables...")
            init_db()
            logger.info("✅ Conversation database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing conversation database: {e}")
            if attempt < max_retries:
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                logger.error("Max retries reached, giving up")
                return False
    
    return False

if __name__ == "__main__":
    logger.info("Starting conversation database initialization")
    success = initialize_db()
    if success:
        logger.info("Database initialization completed successfully")
        sys.exit(0)
    else:
        logger.error("Database initialization failed")
        sys.exit(1) 