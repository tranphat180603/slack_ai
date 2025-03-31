"""
Generate embeddings for Linear data and store them in the PostgreSQL database.
Uses the simplified schema with JSONB data and one embedding per issue.
"""
import os
import logging
import psycopg2
import psycopg2.extras
from psycopg2.extras import execute_values
import json
import tiktoken
import openai
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_embeddings")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    host = os.environ.get("POSTGRES_HOST", "localhost")
    database = os.environ.get("POSTGRES_DB", "linear_rag")
    user = os.environ.get("POSTGRES_USER", "phattran")
    password = os.environ.get("POSTGRES_PASSWORD", "phatdeptrai123")
    
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=60))
def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Get embedding vector for a text using OpenAI API.
    
    Args:
        text: The text to embed
        model: The embedding model to use
        
    Returns:
        Embedding vector as a list of floats
    """
    if not text or text.isspace():
        # Return zero vector for empty text (1536 dimensions for text-embedding-3-small)
        return [0.0] * 1536
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise

def truncate_text_for_embedding(text: str, max_tokens: int = 8000) -> str:
    """
    Truncate text to fit within token limits for embedding models.
    
    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens
        
    Returns:
        Truncated text
    """
    if not text or text.isspace():
        return ""
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate to max_tokens
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def generate_embeddings_for_issues():
    """Generate embeddings for all issues in the database."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Get all issues that don't have embeddings yet
                cur.execute("""
                    SELECT i.id, i.data, i.full_context
                    FROM issues_simplified i
                    LEFT JOIN embeddings_simplified e ON i.id = e.issue_id
                    WHERE e.embedding IS NULL OR e.issue_id IS NULL
                """)
                issues = cur.fetchall()
                
                if not issues:
                    logger.info("No new issues found that need embeddings")
                    return
                
                logger.info(f"Found {len(issues)} issues that need embeddings")
                
                # Process issues in batches
                batch_size = 100
                for i in range(0, len(issues), batch_size):
                    batch = issues[i:i + batch_size]
                    logger.info(f"Processing batch {i//batch_size + 1} of {(len(issues)-1)//batch_size + 1}")
                    
                    # Generate embeddings for this batch
                    embeddings_data = []
                    for issue_id, data_json, full_context in batch:
                        try:
                            # Parse JSON data
                            data = json.loads(data_json) if isinstance(data_json, str) else data_json
                            
                            # Get embedding for the full context
                            embedding = get_embedding(full_context)
                            
                            # Convert data to JSON string if it's a dict
                            data_str = json.dumps(data) if isinstance(data, dict) else data
                            
                            # Add to batch
                            embeddings_data.append((
                                issue_id,
                                full_context,
                                data_str,  # Convert dict to JSON string
                                embedding
                            ))
                        except Exception as e:
                            logger.error(f"Error processing issue {issue_id}: {str(e)}")
                            continue
                    
                    # Insert embeddings in batch
                    if embeddings_data:
                        execute_values(
                            cur,
                            """
                            INSERT INTO embeddings_simplified (issue_id, content, data, embedding)
                            VALUES %s
                            ON CONFLICT (issue_id) DO UPDATE SET
                                content = EXCLUDED.content,
                                data = EXCLUDED.data,
                                embedding = EXCLUDED.embedding
                            """,
                            embeddings_data
                        )
                        conn.commit()
                        logger.info(f"Inserted {len(embeddings_data)} embeddings")
                
                logger.info("Finished generating embeddings")
                
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def main():
    """Run the embedding generation process."""
    parser = argparse.ArgumentParser(description="Generate embeddings for Linear data using simplified schema")
    parser.add_argument('--force', action='store_true', help='Force regeneration of all embeddings')
    
    args = parser.parse_args()
    
    if args.force:
        logger.info("Force regenerating all embeddings")
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("TRUNCATE TABLE embeddings_simplified")
                    conn.commit()
                    logger.info("Cleared all existing embeddings")
        except Exception as e:
            logger.error(f"Error clearing embeddings: {str(e)}")
    
    try:
        print("\nGenerating embeddings for Linear data...")
        generate_embeddings_for_issues()
        
        # Verify final state
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Count issues and embeddings
                cur.execute("SELECT COUNT(*) FROM issues_simplified")
                issues_count = cur.fetchone()[0]
                
                cur.execute("SELECT COUNT(*) FROM embeddings_simplified")
                embeddings_count = cur.fetchone()[0]
                
                # Count embeddings by cycle
                cur.execute("""
                    SELECT data->'cycle'->>'name' as cycle_name, COUNT(*)
                    FROM embeddings_simplified
                    GROUP BY cycle_name
                    ORDER BY cycle_name
                """)
                cycle_counts = cur.fetchall()
                
                print("\nEmbedding Generation Summary:")
                print("=" * 60)
                print(f"Total issues in database: {issues_count}")
                print(f"Total embeddings generated: {embeddings_count}")
                print(f"Coverage: {round(embeddings_count/issues_count*100 if issues_count else 0, 2)}%")
                
                print("\nEmbeddings by cycle:")
                for cycle_name, count in cycle_counts:
                    print(f"- {cycle_name}: {count} embeddings")
                
    except Exception as e:
        logger.error(f"Error in embedding generation: {str(e)}")
        raise

if __name__ == "__main__":
    main() 