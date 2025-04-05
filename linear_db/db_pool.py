"""
Database connection pooling for the Linear RAG system.
Provides a centralized connection pool for all database operations.
"""
import os
import logging
from psycopg2 import pool
from dotenv import load_dotenv
from contextlib import contextmanager

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_pool")

class DatabasePool:

    _pool = None
    
    @classmethod
    def initialize(cls, minconn=1, maxconn=10):
        """Initialize the connection pool."""
        if cls._pool is not None:
            logger.warning("Pool already initialized")
            return
            
        try:
            cls._pool = pool.SimpleConnectionPool(
                minconn,
                maxconn,
                host=os.environ.get("POSTGRES_HOST", "localhost"),
                database=os.environ.get("POSTGRES_DB", "tmai_db"),
                user=os.environ.get("POSTGRES_USER", "phattran"),
                password=os.environ.get("POSTGRES_PASSWORD", "phatdeptrai123"),
                port=5432
            )
            logger.info(f"Initialized connection pool (min={minconn}, max={maxconn})")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            raise
    
    @classmethod
    def get_connection(cls):
        """Get a connection from the pool."""
        if cls._pool is None:
            cls.initialize()
        
        try:
            return cls._pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get connection from pool: {str(e)}")
            raise
    
    @classmethod
    def return_connection(cls, conn):
        """Return a connection to the pool."""
        if cls._pool is None:
            logger.error("Attempting to return connection to uninitialized pool")
            return
            
        try:
            cls._pool.putconn(conn)
        except Exception as e:
            logger.error(f"Failed to return connection to pool: {str(e)}")
            raise
    
    @classmethod
    def close_all(cls):
        """Close all connections in the pool."""
        if cls._pool is not None:
            cls._pool.closeall()
            cls._pool = None
            logger.info("Closed all database connections")

@contextmanager
def get_db_connection():
    """Get a database connection from the pool."""
    conn = None
    try:
        conn = DatabasePool.get_connection()
        yield conn
    finally:
        if conn is not None:
            DatabasePool.return_connection(conn)

# Initialize the pool automatically when the module is imported
DatabasePool.initialize(minconn=1, maxconn=10) 