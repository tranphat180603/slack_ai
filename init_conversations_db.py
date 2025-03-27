"""
Initialize the database tables for Slack AI conversations.
This script creates the tables defined in models.py.
"""
import logging
from database import init_db, check_db_connection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("init_conversations_db")

def main():
    logger.info("Checking database connection...")
    if not check_db_connection():
        logger.error("Database connection failed. Please check your .env file settings.")
        return
    
    logger.info("Initializing conversation tables...")
    try:
        init_db()
        logger.info("✅ Conversation tables created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating conversation tables: {str(e)}")

if __name__ == "__main__":
    main() 