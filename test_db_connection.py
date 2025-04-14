#!/usr/bin/env python3
"""
Simple script to test database connection and initialization
"""

import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_test")

# Load environment variables
load_dotenv()

def test_db_connection():
    """Test connection to the database using environment variables"""
    # Get database connection details from environment
    db_host = os.environ.get("POSTGRES_HOST", "db")
    db_name = os.environ.get("POSTGRES_DB", "tmai_db")
    db_user = os.environ.get("POSTGRES_USER", "phattran")
    db_password = os.environ.get("POSTGRES_PASSWORD", "phatdeptrai123")
    
    # Build connection string
    conn_string = f"host={db_host} dbname={db_name} user={db_user} password={db_password}"
    logger.info(f"Testing connection to: host={db_host} dbname={db_name} user={db_user}")
    
    try:
        # Connect to the database
        conn = psycopg2.connect(conn_string)
        
        # Test query
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            db_version = cur.fetchone()
            logger.info(f"Connected successfully to PostgreSQL version: {db_version[0]}")
            
            # List all tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema='public'
            """)
            tables = cur.fetchall()
            if tables:
                logger.info("Found tables:")
                for table in tables:
                    logger.info(f"  - {table[0]}")
            else:
                logger.warning("No tables found in database")
                
        conn.close()
        logger.info("Database connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Starting database connection test...")
    try:
        success = test_db_connection()
        if success:
            print("✅ Database connection test passed")
            sys.exit(0)
        else:
            print("❌ Database connection test failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        sys.exit(1) 