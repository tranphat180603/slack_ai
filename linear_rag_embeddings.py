"""
Generate embeddings for Linear data and store them in the PostgreSQL database.
"""
import os
import logging
import psycopg2
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
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "Phatdeptrai@123")
    
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

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of tokens with overlap.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum number of tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or text.isspace():
        return []
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    chunks = []
    i = 0
    while i < len(tokens):
        # Get chunk with specified size
        chunk_end = min(i + chunk_size, len(tokens))
        chunk = encoding.decode(tokens[i:chunk_end])
        chunks.append(chunk)
        
        # Move to next chunk, considering overlap
        i += (chunk_size - overlap)
        
    return chunks

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

def get_issue_metadata(issue_id: str, conn) -> Dict[str, Any]:
    """
    Get metadata for an issue to store with its embedding.
    
    Args:
        issue_id: The ID of the issue
        conn: Database connection
        
    Returns:
        Dictionary of metadata
    """
    cur = conn.cursor()
    
    try:
        cur.execute("""
        SELECT 
            i.title, 
            i.state, 
            i.priority,
            i.created_at,
            i.completed_at,
            t.name as team_name, 
            t.key as team_key,
            e.name as assignee_name,
            c.name as cycle_name,
            c.number as cycle_number,
            p.title as parent_title
        FROM 
            issues i
        LEFT JOIN 
            teams t ON i.team_id = t.id
        LEFT JOIN 
            employees e ON i.assignee_id = e.id
        LEFT JOIN 
            cycles c ON i.cycle_id = c.id
        LEFT JOIN 
            issues p ON i.parent_id = p.id
        WHERE 
            i.id = %s
        """, (issue_id,))
        
        row = cur.fetchone()
        if not row:
            return {}
        
        return {
            "title": row[0],
            "state": row[1],
            "priority": row[2],
            "created_at": row[3].isoformat() if row[3] else None,
            "completed_at": row[4].isoformat() if row[4] else None,
            "team_name": row[5],
            "team_key": row[6],
            "assignee_name": row[7],
            "cycle_name": row[8],
            "cycle_number": row[9],
            "parent_title": row[10],
            "has_children": has_children(issue_id, conn)
        }
    finally:
        cur.close()

def has_children(issue_id: str, conn) -> bool:
    """Check if an issue has child issues."""
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM issues WHERE parent_id = %s", (issue_id,))
        count = cur.fetchone()[0]
        return count > 0
    finally:
        cur.close()

def generate_issue_embeddings():
    """Generate embeddings for all issues."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get all issues
        cur.execute("""
        SELECT 
            id, title, description, team_id, assignee_id, cycle_id 
        FROM 
            issues
        """)
        
        issues = cur.fetchall()
        logger.info(f"Processing embeddings for {len(issues)} issues")
        
        issue_count = 0
        embedding_count = 0
        
        for issue in issues:
            issue_id, title, description, team_id, assignee_id, cycle_id = issue
            
            # Get metadata
            metadata = get_issue_metadata(issue_id, conn)
            
            # Generate embedding for title
            title_with_context = f"Issue: {title}\nTeam: {metadata.get('team_name', '')}\nAssignee: {metadata.get('assignee_name', '')}\nState: {metadata.get('state', '')}"
            title_embedding = get_embedding(title_with_context)
            
            # Store title embedding
            cur.execute(
                """
                INSERT INTO embeddings
                (source_type, source_id, cycle_id, team_id, assignee_id, content, embedding, metadata, chunk_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (source_id, source_type, chunk_index)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP
                """,
                ('issue_title', issue_id, cycle_id, team_id, assignee_id, title_with_context, 
                 title_embedding, json.dumps(metadata), 0)
            )
            embedding_count += 1
            
            # Generate embeddings for description
            if description and not description.isspace():
                chunks = chunk_text(description)
                for i, chunk in enumerate(chunks):
                    chunk_embedding = get_embedding(chunk)
                    cur.execute(
                        """
                        INSERT INTO embeddings
                        (source_type, source_id, cycle_id, team_id, assignee_id, content, embedding, metadata, chunk_index)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                        ON CONFLICT (source_id, source_type, chunk_index)
                        DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            metadata = EXCLUDED.metadata,
                            created_at = CURRENT_TIMESTAMP
                        """,
                        ('issue_description', issue_id, cycle_id, team_id, assignee_id, chunk, 
                         chunk_embedding, json.dumps(metadata), i)
                    )
                    embedding_count += 1
            
            issue_count += 1
            if issue_count % 10 == 0:
                conn.commit()
                logger.info(f"Processed {issue_count}/{len(issues)} issues, {embedding_count} embeddings")
        
        conn.commit()
        logger.info(f"Completed processing {issue_count} issues, {embedding_count} embeddings")
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error generating embeddings for issues: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def generate_comment_embeddings():
    """Generate embeddings for all comments."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get all comments with issue info
        cur.execute("""
        SELECT 
            c.id, c.body, c.created_at,
            i.id as issue_id, i.title as issue_title,
            i.team_id, i.assignee_id, i.cycle_id
        FROM 
            comments c
        JOIN 
            issues i ON c.issue_id = i.id
        """)
        
        comments = cur.fetchall()
        logger.info(f"Processing embeddings for {len(comments)} comments")
        
        comment_count = 0
        embedding_count = 0
        
        for comment in comments:
            comment_id, body, created_at, issue_id, issue_title, team_id, assignee_id, cycle_id = comment
            
            if not body or body.isspace():
                continue
            
            # Get issue metadata
            metadata = get_issue_metadata(issue_id, conn)
            metadata.update({
                "comment_created_at": created_at.isoformat() if created_at else None,
                "issue_id": issue_id,
                "issue_title": issue_title
            })
            
            # Generate embeddings for comment
            chunks = chunk_text(body)
            for i, chunk in enumerate(chunks):
                chunk_embedding = get_embedding(chunk)
                cur.execute(
                    """
                    INSERT INTO embeddings
                    (source_type, source_id, cycle_id, team_id, assignee_id, content, embedding, metadata, chunk_index)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                    ON CONFLICT (source_id, source_type, chunk_index)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = CURRENT_TIMESTAMP
                    """,
                    ('comment', comment_id, cycle_id, team_id, assignee_id, chunk, 
                     chunk_embedding, json.dumps(metadata), i)
                )
                embedding_count += 1
            
            comment_count += 1
            if comment_count % 10 == 0:
                conn.commit()
                logger.info(f"Processed {comment_count}/{len(comments)} comments, {embedding_count} embeddings")
        
        conn.commit()
        logger.info(f"Completed processing {comment_count} comments, {embedding_count} embeddings")
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error generating embeddings for comments: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def generate_employee_cycle_summaries():
    """
    Generate summary embeddings for each employee's work in each cycle.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get all employee-cycle combinations
        cur.execute("""
        SELECT DISTINCT
            e.id as employee_id,
            e.name as employee_name,
            c.id as cycle_id,
            c.name as cycle_name,
            t.id as team_id,
            t.name as team_name
        FROM 
            issues i
        JOIN
            employees e ON i.assignee_id = e.id
        JOIN
            cycles c ON i.cycle_id = c.id
        JOIN
            teams t ON i.team_id = t.id
        """)
        
        employee_cycles = cur.fetchall()
        logger.info(f"Found {len(employee_cycles)} employee-cycle combinations")
        
        for ec in employee_cycles:
            employee_id, employee_name, cycle_id, cycle_name, team_id, team_name = ec
            
            logger.info(f"Generating summary for {employee_name} in {cycle_name}")
            
            # Get all issues for this employee in this cycle
            cur.execute("""
            SELECT 
                id, title, description, state, priority, completed_at, created_at
            FROM 
                issues
            WHERE 
                assignee_id = %s AND cycle_id = %s
            ORDER BY
                priority DESC, created_at
            """, (employee_id, cycle_id))
            
            issues = cur.fetchall()
            
            if not issues:
                logger.info(f"No issues found for {employee_name} in {cycle_name}")
                continue
            
            # Generate summary text
            summary_text = f"Work summary for {employee_name} in {cycle_name} ({team_name}):\n\n"
            
            completed_issues = 0
            total_issues = len(issues)
            
            for issue in issues:
                issue_id, title, description, state, priority, completed_at, created_at = issue
                status = "✓" if completed_at else "□"
                priority_str = f"P{priority}" if priority is not None else ""
                
                summary_text += f"{status} {title} ({state}) {priority_str}\n"
                if completed_at:
                    completed_issues += 1
            
            completion_rate = (completed_issues / total_issues) * 100 if total_issues > 0 else 0
            summary_text += f"\nCompletion rate: {completion_rate:.1f}% ({completed_issues}/{total_issues} issues)"
            
            # Generate metadata
            metadata = {
                "employee_name": employee_name,
                "cycle_name": cycle_name,
                "team_name": team_name,
                "total_issues": total_issues,
                "completed_issues": completed_issues,
                "completion_rate": completion_rate,
                "summary_type": "employee_cycle"
            }
            
            # Generate embedding for the summary
            summary_embedding = get_embedding(summary_text)
            
            # Store summary embedding
            source_id = f"{employee_id}_{cycle_id}"
            cur.execute(
                """
                INSERT INTO embeddings
                (source_type, source_id, cycle_id, team_id, assignee_id, content, embedding, metadata, chunk_index)
                VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
                ON CONFLICT (source_id, source_type, chunk_index)
                DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    created_at = CURRENT_TIMESTAMP
                """,
                ('employee_cycle_summary', source_id, cycle_id, team_id, employee_id, summary_text,
                 summary_embedding, json.dumps(metadata), 0)
            )
        
        conn.commit()
        logger.info(f"Generated summaries for {len(employee_cycles)} employee-cycle combinations")
    
    except Exception as e:
        conn.rollback()
        logger.error(f"Error generating employee-cycle summaries: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def main():
    """Run the embedding generation process."""
    parser = argparse.ArgumentParser(description="Generate embeddings for Linear data")
    parser.add_argument("--issues", action="store_true", help="Generate embeddings for issues")
    parser.add_argument("--comments", action="store_true", help="Generate embeddings for comments")
    parser.add_argument("--summaries", action="store_true", help="Generate employee-cycle summaries")
    parser.add_argument("--all", action="store_true", help="Generate all types of embeddings")
    
    args = parser.parse_args()
    
    if args.all or (not args.issues and not args.comments and not args.summaries):
        # Default: run all
        print("Generating embeddings for all data types...")
        generate_issue_embeddings()
        generate_comment_embeddings()
        generate_employee_cycle_summaries()
    else:
        if args.issues:
            print("Generating embeddings for issues...")
            generate_issue_embeddings()
        
        if args.comments:
            print("Generating embeddings for comments...")
            generate_comment_embeddings()
        
        if args.summaries:
            print("Generating employee-cycle summaries...")
            generate_employee_cycle_summaries()
    
    print("✓ Embedding generation completed successfully")

if __name__ == "__main__":
    main() 