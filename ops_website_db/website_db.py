"""
Database module for website content storage with PostgreSQL and pgvector.
Creates a table to store website content and embeddings for RAG systems.
"""
import os
import sys
import logging
import json
import psycopg2
from typing import Dict, List, Any, Optional, Tuple, Iterator, Generator
from dotenv import load_dotenv
from ops_linear_db.linear_rag_embeddings import get_embedding

# Fix imports when running as a script
if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from ops_linear_db.db_pool import get_db_connection
else:
    from ops_linear_db.db_pool import get_db_connection

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("website_db")

class WebsiteDB:
    """Database interface for website content storage."""
    
    WEBSITE_TYPES = ["main", "research", "blog"]
    
    def __init__(self):
        """Initialize the database interface."""
        self.initialize_database()
    
    def initialize_database(self):
        """Create the database schema for website content storage."""
        logger.info("Initializing website database...")
        
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Enable pgvector extension if not already enabled
                    logger.info("Ensuring pgvector extension is enabled...")
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    
                    # Create enum type for website types
                    logger.info("Creating website_type enum...")
                    cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'website_type') THEN
                            CREATE TYPE website_type AS ENUM ('main', 'research', 'blog');
                        END IF;
                    END$$;
                    """)
                    
                    # Create the website_content table
                    logger.info("Creating website_content table...")
                    cur.execute("""
                    CREATE TABLE IF NOT EXISTS website_content (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        website_type website_type NOT NULL,
                        url TEXT NOT NULL,
                        title TEXT,
                        full_content TEXT,
                        content_chunk TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        embedding vector(1536),
                        metadata JSONB NOT NULL DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(url, chunk_index)
                    );
                    """)
                    
                    # Create indices
                    logger.info("Creating indices...")
                    
                    # Index for URL for fast lookups
                    cur.execute("""
                    CREATE INDEX IF NOT EXISTS website_content_url_idx 
                    ON website_content (url);
                    """)
                    
                    # Index for website type
                    cur.execute("""
                    CREATE INDEX IF NOT EXISTS website_content_type_idx 
                    ON website_content (website_type);
                    """)
                    
                    # Text search index
                    cur.execute("""
                    CREATE INDEX IF NOT EXISTS website_content_text_idx 
                    ON website_content USING GIN (to_tsvector('english', content_chunk));
                    """)
                    
                    # Vector similarity index using HNSW for fast similarity search
                    cur.execute("""
                    CREATE INDEX IF NOT EXISTS website_content_hnsw_idx
                    ON website_content
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
                    DROP TRIGGER IF EXISTS update_website_content_modtime ON website_content;
                    """)
                    
                    cur.execute("""
                    CREATE TRIGGER update_website_content_modtime
                    BEFORE UPDATE ON website_content
                    FOR EACH ROW
                    EXECUTE FUNCTION update_modified_column();
                    """)
                    
                    conn.commit()
                    logger.info("Website database schema initialized successfully")
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error initializing database: {str(e)}")
                    raise
    
    def get_url_record(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Check if a URL already exists in the database.
        
        Args:
            url: The URL to check
            
        Returns:
            Dictionary with URL info or None if not found
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                    SELECT url, website_type, MAX(updated_at) as last_updated
                    FROM website_content
                    WHERE url = %s
                    GROUP BY url, website_type
                    """, (url,))
                    
                    result = cur.fetchone()
                    if result:
                        return {
                            "url": result[0],
                            "website_type": result[1],
                            "last_updated": result[2]
                        }
                    return None
                    
                except Exception as e:
                    logger.error(f"Error checking URL existence: {str(e)}")
                    return None
    
    def store_page_chunks(
        self,
        website_type: str, 
        url: str, 
        title: str, 
        full_content: str, 
        chunks: List[str],
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Store content chunks and their embeddings for a webpage.
        """
        if website_type not in self.WEBSITE_TYPES:
            logger.error(f"Invalid website type: {website_type}")
            return False
            
        metadata_json = json.dumps(metadata or {})
        total_chunks = len(chunks)
        
        try:
            with get_db_connection() as conn:
                conn.autocommit = False
                with conn.cursor() as cur:
                    # Remove any existing batch of chunks for this URL
                    cur.execute("DELETE FROM website_content WHERE url = %s", (url,))
                    
                    batch_size = 20
                    for i in range(0, total_chunks, batch_size):
                        batch_end = min(i + batch_size, total_chunks)
                        for j, chunk in enumerate(chunks[i:batch_end]):
                            idx = i + j
                            # Only store full_content on first chunk
                            current_full = full_content if idx == 0 else None
                            
                            # **NEW**: compute embedding
                            try:
                                embedding = get_embedding(chunk)
                            except Exception as e:
                                logger.error(f"Embedding failed for chunk {idx}: {e}")
                                embedding = None
                            
                            cur.execute("""
                                INSERT INTO website_content
                                  (website_type, url, title, full_content,
                                   content_chunk, chunk_index, embedding, metadata)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                website_type,
                                url,
                                title,
                                current_full,
                                chunk,
                                idx,
                                embedding,
                                metadata_json
                            ))
                        
                        conn.commit()
                        # free up memory between batches
                        import gc; gc.collect()
                    
                    logger.info(f"Stored {total_chunks} chunks (with embeddings) for URL: {url}")
                    return True

        except Exception as e:
            logger.error(f"Error storing page chunks: {e}")
            return False

    
    def update_embedding(self, chunk_id: str, embedding: List[float]) -> bool:
        """
        Update the embedding for a specific content chunk.
        
        Args:
            chunk_id: UUID of the chunk
            embedding: Vector embedding as a list of floats
            
        Returns:
            True if successful, False otherwise
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute("""
                    UPDATE website_content
                    SET embedding = %s
                    WHERE id = %s
                    """, (embedding, chunk_id))
                    
                    conn.commit()
                    return True
                    
                except Exception as e:
                    conn.rollback()
                    logger.error(f"Error updating embedding: {str(e)}")
                    return False
    
    def search_website_content(
        self, 
        query: str = None, 
        website_type: str = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search website_content, filtering by type or URL if given,
        and ordering by vector similarity if `query` is given.
        """
        # 1) If we have a text query, compute its embedding and
        #    format it as a pgvector literal
        vector_literal = None
        if query is not None:
            emb = get_embedding(query)  # List[float]
            # build a string like "[0.1,0.2,0.3,…]"
            vector_literal = "[" + ",".join(f"{x:.6f}" for x in emb) + "]"

        # 2) Build dynamic SELECT / SIMILARITY expression
        if vector_literal is not None:
            sim_select = "1 - (embedding <=> %s::vector) AS similarity"
        else:
            sim_select = "NULL AS similarity"

        select_sql = f"""
            SELECT id, url, title, content_chunk, metadata,
                   {sim_select}
            FROM website_content
        """

        # 3) Build WHERE clauses & params
        where_clauses = []
        params: List[Any] = []

        if vector_literal is not None:
            where_clauses.append("embedding IS NOT NULL")
            # this parameter is only for the SELECT-similarity
            params.append(vector_literal)

        if website_type:
            if website_type not in self.WEBSITE_TYPES:
                raise ValueError(f"Invalid website_type: {website_type}")
            where_clauses.append("website_type = %s")
            params.append(website_type)

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        # 4) ORDER BY only if we have a vector
        order_sql = ""
        if vector_literal is not None:
            order_sql = "ORDER BY embedding <=> %s::vector"
            # note: append again for ORDER BY
            params.append(vector_literal)

        # 5) Finally, LIMIT
        limit_sql = "LIMIT %s"
        params.append(limit)

        sql = "\n".join([select_sql, where_sql, order_sql, limit_sql])

        # 6) Execute
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        # 7) Map to dict
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "url": row[1],
                "title": row[2],
                "content": row[3],
                "metadata": row[4],
                "similarity": row[5],
            })
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the website content database.
        
        Returns:
            Dictionary with database statistics
        """
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    stats = {}
                    
                    # Get total count
                    cur.execute("SELECT COUNT(*) FROM website_content;")
                    stats["total_chunks"] = cur.fetchone()[0]
                    
                    # Get count by website type
                    cur.execute("""
                    SELECT website_type, COUNT(*) 
                    FROM website_content 
                    GROUP BY website_type;
                    """)
                    stats["counts_by_type"] = {row[0]: row[1] for row in cur.fetchall()}
                    
                    # Get unique URLs count
                    cur.execute("SELECT COUNT(DISTINCT url) FROM website_content;")
                    stats["unique_urls"] = cur.fetchone()[0]
                    
                    # Get count of chunks with embeddings
                    cur.execute("SELECT COUNT(*) FROM website_content WHERE embedding IS NOT NULL;")
                    stats["chunks_with_embeddings"] = cur.fetchone()[0]
                    
                    # Get newest record date
                    cur.execute("SELECT MAX(created_at) FROM website_content;")
                    stats["newest_record"] = cur.fetchone()[0]
                    
                    return stats
                    
                except Exception as e:
                    logger.error(f"Error getting database stats: {str(e)}")
                    return {"error": str(e)}

def check_database_status():
    """Check the status of the website database and report stats."""
    try:
        db = WebsiteDB()
        stats = db.get_stats()
        
        print("\nWebsite Database Status:")
        print(f"Total Chunks: {stats.get('total_chunks', 'N/A')}")
        print(f"Unique URLs: {stats.get('unique_urls', 'N/A')}")
        print(f"Chunks with Embeddings: {stats.get('chunks_with_embeddings', 'N/A')}")
        
        if "counts_by_type" in stats:
            print("\nChunks by Website Type:")
            for website_type, count in stats["counts_by_type"].items():
                print(f"- {website_type}: {count}")
        
        if "newest_record" in stats:
            print(f"\nNewest Record: {stats['newest_record']}")
        
        return stats
    except Exception as e:
        print(f"Error checking database status: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    try:
        # Initialize the database
        db = WebsiteDB()
        print("✓ Website database schema created successfully")
        
        # Check and report on the database status
        check_database_status()
        
        # Search for content
        results = db.search_website_content()
        print(results)
    except Exception as e:
        print(f"✗ Error creating database schema: {str(e)}")
