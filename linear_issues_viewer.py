"""
Display titles and descriptions from the issues table in the Linear RAG database.
"""
import os
import psycopg2
import pandas as pd
from tabulate import tabulate
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_issues_viewer")

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

def get_issues_title_description(limit=20):
    """
    Get titles and descriptions from the issues table.
    
    Args:
        limit: Maximum number of issues to retrieve
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Get total count
        cur.execute("SELECT COUNT(*) FROM issues")
        total_count = cur.fetchone()[0]
        print(f"Total issues in database: {total_count}")
        
        # Get issues with title and description
        cur.execute(f"""
        SELECT id, title, description 
        FROM issues 
        ORDER BY created_at DESC
        LIMIT {limit}
        """)
        
        rows = cur.fetchall()
        
        # Process results
        issues = []
        for row in rows:
            issue_id, title, description = row
            
            # Truncate description if needed
            if description and len(description) > 200:
                description = description[:200] + "..."
            
            issues.append({
                "id": issue_id,
                "title": title,
                "description": description
            })
        
        return issues
    except Exception as e:
        logger.error(f"Error retrieving issues: {str(e)}")
        print(f"Error retrieving issues: {str(e)}")
        return []
    finally:
        cur.close()
        conn.close()

def main():
    issues = get_issues_title_description()
    
    if not issues:
        print("No issues found or an error occurred.")
        return
    
    # Create DataFrame for better display
    df = pd.DataFrame(issues)
    
    # Display the data
    print("\n=== Issues (Title and Description) ===")
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=True))

if __name__ == "__main__":
    main() 