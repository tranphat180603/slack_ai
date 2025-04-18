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
from bs4 import BeautifulSoup
import markdown

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

    def clean_text(self, text: str) -> str:
        # 1. Convert Markdown to HTML
        html = markdown.markdown(text, extensions=['extra', 'smarty'])
        # 2. Strip HTML as above
        soup = BeautifulSoup(html, 'html.parser')
        return ' '.join(soup.get_text(separator=' ').split())
    
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
        query: Optional[str] = None,
        website_type: Optional[str] = None,
        distinct_on_url: bool = False,
        return_full_content: bool = False,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search website_content.

        Parameters
        ----------
        query : str | None
            If provided, a text query that will be embedded and used for vector
            similarity search.
        website_type : str | None
            Optional filter on the website_type enum column.
        distinct_on_url : bool
            When True, collapse results so you get at most one row per URL.
        return_full_content : bool
            When True, include the 'full_content' column in SELECT and in the
            returned dicts (only present for the chunk that originally stored it).
        limit : int
            Maximum number of rows to return *after* DISTINCT ON (if enabled).
        """
        # ------------------------------------------------------------------ #
        # 1. Build embedding for the query (if any)
        # ------------------------------------------------------------------ #
        vector_literal: Optional[str] = None
        if query:
            embedding = get_embedding(query)  # -> List[float]
            vector_literal = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

        # ------------------------------------------------------------------ #
        # 2. Similarity select fragment
        # ------------------------------------------------------------------ #
        similarity_expr = (
            "1 - (embedding <=> %s::vector) AS similarity"
            if vector_literal is not None
            else "NULL AS similarity"
        )

        # ------------------------------------------------------------------ #
        # 3. SELECT clause
        # ------------------------------------------------------------------ #
        column_list = [
            "id",
            "url",
            "title",
        ]
        if return_full_content:
            column_list.append("full_content")
        column_list.extend(["content_chunk", "metadata"])

        select_columns = ", ".join(column_list) + f", {similarity_expr}"

        if distinct_on_url:
            select_sql = f"SELECT DISTINCT ON (url) {select_columns} FROM website_content"
        else:
            select_sql = f"SELECT {select_columns} FROM website_content"

        # ------------------------------------------------------------------ #
        # 4. WHERE clause
        # ------------------------------------------------------------------ #
        where_parts: List[str] = []
        params: List[Any] = []

        if vector_literal is not None:
            where_parts.append("embedding IS NOT NULL")
            params.append(vector_literal)  # for the similarity expression

        if website_type:
            if website_type not in self.WEBSITE_TYPES:
                raise ValueError(f"Invalid website_type: {website_type}")
            where_parts.append("website_type = %s")
            params.append(website_type)

        where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""

        # ------------------------------------------------------------------ #
        # 5. ORDER BY clause
        # ------------------------------------------------------------------ #
        if distinct_on_url:
            if vector_literal is not None:
                order_sql = "ORDER BY url, embedding <=> %s::vector"
                params.append(vector_literal)  # second time for ORDER BY
            else:
                order_sql = "ORDER BY url, updated_at DESC"
        else:
            if vector_literal is not None:
                order_sql = "ORDER BY embedding <=> %s::vector"
                params.append(vector_literal)  # second time for ORDER BY
            else:
                order_sql = ""
        
        # ------------------------------------------------------------------ #
        # 6. LIMIT
        # ------------------------------------------------------------------ #
        limit_sql = "LIMIT %s"
        params.append(limit)

        # ------------------------------------------------------------------ #
        # 7. Assemble final SQL
        # ------------------------------------------------------------------ #
        sql = "\n".join(
            part for part in (select_sql, where_sql, order_sql, limit_sql) if part
        )

        # ------------------------------------------------------------------ #
        # 8. Execute
        # ------------------------------------------------------------------ #
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        # ------------------------------------------------------------------ #
        # 9. Map results back to Python dicts
        # ------------------------------------------------------------------ #
        results: List[Dict[str, Any]] = []
        for row in rows:
            idx = 0
            result = {
                "id": row[idx],
                "url": row[idx + 1],
                "title": row[idx + 2],
            }
            idx += 3

            if return_full_content:
                raw_full = row[idx]
                result["full_content"] = (
                    self.clean_text(raw_full) if raw_full is not None else None
                )
                idx += 1


            result.update(
                {
                    "content": row[idx],
                    "metadata": row[idx + 1],
                    "similarity": row[idx + 2],
                }
            )
            results.append(result)

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
        results = db.search_website_content(query="Token Metrics API", distinct_on_url=True, return_full_content=True, limit=10, website_type="main")
        for result in results:
            print(result)
            print("--------------------------------")
    except Exception as e:
        print(f"✗ Error creating database schema: {str(e)}")
