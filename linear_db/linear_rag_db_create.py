"""
Database setup script for Linear RAG system with PostgreSQL and pgvector.
Creates a single table to store embeddings for various Linear object types.
"""
import os
import sys
import logging
import psycopg2
from dotenv import load_dotenv

# Fix imports when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from linear_db.db_pool import get_db_connection
else:
    from .db_pool import get_db_connection

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_db")

def initialize_database():
    """Create the simplified database schema for the Linear RAG system."""
    logger.info("Initializing database...")
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Enable pgvector extension
                logger.info("Enabling pgvector extension...")
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create the linear_embeddings table
                logger.info("Creating linear_embeddings table...")
                cur.execute("""
                CREATE TABLE IF NOT EXISTS linear_embeddings (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    object_type VARCHAR(50) NOT NULL,
                    object_id VARCHAR(100) NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB NOT NULL DEFAULT '{}',
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(object_type, object_id)
                );
                """)
                
                # Create indices
                logger.info("Creating indices...")
                
                # Index for object type and ID combination for uniqueness and fast lookups
                cur.execute("""
                CREATE INDEX IF NOT EXISTS linear_embeddings_object_idx 
                ON linear_embeddings (object_type, object_id);
                """)
                
                # Index for team key which will be used in most queries
                cur.execute("""
                CREATE INDEX IF NOT EXISTS linear_embeddings_team_idx 
                ON linear_embeddings ((metadata->>'team_key'));
                """)
                
                # Index on issue number for comments and issues
                cur.execute("""
                CREATE INDEX IF NOT EXISTS linear_embeddings_issue_number_idx 
                ON linear_embeddings ((metadata->>'issue_number'));
                """)
                
                # Index on cycle for issues
                cur.execute("""
                CREATE INDEX IF NOT EXISTS linear_embeddings_cycle_idx 
                ON linear_embeddings ((metadata->>'cycle_number'));
                """)
                
                # Vector similarity index using HNSW for fast similarity search
                cur.execute("""
                CREATE INDEX IF NOT EXISTS linear_embeddings_hnsw_idx
                ON linear_embeddings
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64);
                """)
                
                # Create a function for automatically updating the updated_at timestamp
                cur.execute("""
                CREATE OR REPLACE FUNCTION update_modified_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
                """)
                
                # Create a trigger for automatically updating the updated_at timestamp
                cur.execute("""
                DROP TRIGGER IF EXISTS update_linear_embeddings_modtime ON linear_embeddings;
                """)
                
                cur.execute("""
                CREATE TRIGGER update_linear_embeddings_modtime
                BEFORE UPDATE ON linear_embeddings
                FOR EACH ROW
                EXECUTE FUNCTION update_modified_column();
                """)
                
                conn.commit()
                logger.info("Database schema initialized successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error initializing database: {str(e)}")
                raise

def check_database_status():
    """Check the status of the database and report on existing data."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Check if the table exists
                cur.execute("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_name = 'linear_embeddings'
                );
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    return {"status": "Table does not exist"}
                
                # Get counts of each object type
                cur.execute("""
                SELECT object_type, COUNT(*) 
                FROM linear_embeddings 
                GROUP BY object_type
                ORDER BY object_type;
                """)
                
                type_counts = cur.fetchall()
                
                # Get total count
                cur.execute("SELECT COUNT(*) FROM linear_embeddings;")
                total_count = cur.fetchone()[0]
                
                # Get creation date of the oldest record
                cur.execute("SELECT MIN(created_at) FROM linear_embeddings;")
                oldest_record = cur.fetchone()[0]
                
                # Get creation date of the newest record
                cur.execute("SELECT MAX(created_at) FROM linear_embeddings;")
                newest_record = cur.fetchone()[0]
                
                return {
                    "status": "Table exists",
                    "total_records": total_count,
                    "type_counts": type_counts,
                    "oldest_record": oldest_record,
                    "newest_record": newest_record
                }
                
            except Exception as e:
                logger.error(f"Error checking database status: {str(e)}")
                return {"status": "Error", "message": str(e)}

if __name__ == "__main__":
    try:
        # Initialize the database
        initialize_database()
        print("✓ Linear RAG database schema created successfully")
        
        # Check and report on the database status
        status = check_database_status()
        
        if status["status"] == "Table exists":
            print("\nDatabase Status:")
            print(f"Total Records: {status['total_records']}")
            
            if status['type_counts']:
                print("\nRecords by Type:")
                for object_type, count in status['type_counts']:
                    print(f"- {object_type}: {count}")
            
            if status['oldest_record']:
                print(f"\nOldest Record: {status['oldest_record']}")
                print(f"Newest Record: {status['newest_record']}")
        
    except Exception as e:
        print(f"✗ Error creating database schema: {str(e)}") 