"""
Database setup script for Linear RAG system with PostgreSQL and pgvector.
Creates the database schema, tables, and indices needed.
"""
import os
import psycopg2
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_db")

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    database = os.environ.get("POSTGRES_DB", "linear_rag")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "Phatdeptrai@123")
    
    logger.info(f"Connecting to PostgreSQL database: {database} on {host}")
    
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        logger.info("Database connection established successfully")
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise

def initialize_database():
    """Create the database schema for the Linear RAG system."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Enable pgvector extension
        logger.info("Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create teams table
        logger.info("Creating teams table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS teams (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            key TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create cycles table
        logger.info("Creating cycles table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS cycles (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            number INTEGER,
            team_id TEXT REFERENCES teams(id),
            starts_at TIMESTAMP,
            ends_at TIMESTAMP,
            progress FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create employees table
        logger.info("Creating employees table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            team_id TEXT REFERENCES teams(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create issues table
        logger.info("Creating issues table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            state TEXT,
            state_type TEXT,
            cycle_id TEXT REFERENCES cycles(id),
            assignee_id TEXT REFERENCES employees(id),
            team_id TEXT REFERENCES teams(id),
            priority INTEGER,
            estimate FLOAT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            completed_at TIMESTAMP,
            parent_id TEXT REFERENCES issues(id) NULL
        );
        """)
        
        # Create comments table
        logger.info("Creating comments table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS comments (
            id TEXT PRIMARY KEY,
            issue_id TEXT REFERENCES issues(id),
            user_id TEXT REFERENCES employees(id),
            body TEXT NOT NULL,
            created_at TIMESTAMP
        );
        """)
        
        # Create embeddings table with 1536 dimensions for text-embedding-3-small
        logger.info("Creating embeddings table...")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            source_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            cycle_id TEXT REFERENCES cycles(id),
            team_id TEXT REFERENCES teams(id),
            assignee_id TEXT REFERENCES employees(id),
            content TEXT NOT NULL,
            embedding vector(1536),
            metadata JSONB,
            chunk_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_id, source_type, chunk_index)
        );
        """)
        
        # Create indices
        logger.info("Creating indices...")
        
        # Vector similarity index using HNSW (Hierarchical Navigable Small World)
        cur.execute("""
        CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
        ON embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64);
        """)
        
        conn.commit()
        logger.info("Database schema initialized successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    try:
        initialize_database()
        print("✓ Linear RAG database schema created successfully")
    except Exception as e:
        print(f"✗ Error creating database schema: {str(e)}") 