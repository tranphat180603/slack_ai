"""
Database setup script for Linear RAG system with PostgreSQL and pgvector.
Creates the database schema, tables, and indices needed.
"""
import os
import logging
from dotenv import load_dotenv
from db_pool import DatabasePool, get_db_connection

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_db")

def initialize_database():
    """Create the database schema for the Linear RAG system."""
    # Initialize the connection pool
    DatabasePool.initialize(minconn=1, maxconn=10)
    logger.info("Database connection pool initialized")
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Enable pgvector extension
                logger.info("Enabling pgvector extension...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create simplified issues table
                logger.info("Creating issues_simplified table...")
                cur.execute("""
                CREATE TABLE IF NOT EXISTS issues_simplified (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL,
                    full_context TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create indices for the issues table
                cur.execute("""
                CREATE INDEX IF NOT EXISTS issues_simplified_team_idx 
                ON issues_simplified ((data->'team'->>'key'));
                """)
                
                cur.execute("""
                CREATE INDEX IF NOT EXISTS issues_simplified_cycle_idx 
                ON issues_simplified ((data->'cycle'->>'name'));
                """)
                
                cur.execute("""
                CREATE INDEX IF NOT EXISTS issues_simplified_assignee_idx 
                ON issues_simplified ((data->'assignee'->>'name'));
                """)
                
                # Create simplified embeddings table
                logger.info("Creating embeddings_simplified table...")
                cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings_simplified (
                    id SERIAL PRIMARY KEY,
                    issue_id TEXT REFERENCES issues_simplified(id) UNIQUE,
                    embedding vector(1536),
                    content TEXT NOT NULL,
                    data JSONB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # Create vector similarity index using HNSW
                logger.info("Creating vector similarity index...")
                cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_simplified_hnsw_idx
                ON embeddings_simplified
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """)
                
                # Create indices for common filters on embeddings table
                cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_simplified_team_idx 
                ON embeddings_simplified ((data->'team'->>'key'));
                """)
                
                cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_simplified_cycle_idx 
                ON embeddings_simplified ((data->'cycle'->>'name'));
                """)
                
                cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_simplified_assignee_idx 
                ON embeddings_simplified ((data->'assignee'->>'name'));
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error initializing database: {str(e)}")
                raise

if __name__ == "__main__":
    try:
        initialize_database()
        print("✓ Linear RAG database schema created successfully")
    except Exception as e:
        print(f"✗ Error creating database schema: {str(e)}")
    finally:
        DatabasePool.close_all() 