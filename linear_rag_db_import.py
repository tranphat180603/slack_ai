"""
Import data from Linear JSON exports into the PostgreSQL database with the simplified schema.
"""
import os
import json
import logging
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse
from dotenv import load_dotenv
from db_pool import DatabasePool, get_db_connection

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_import")

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

def create_full_context(issue_data):
    """Create a human-readable full context string from issue data."""
    
    # Basic issue information
    context = f"Issue: {issue_data.get('title', '')}\n"
    context += f"State: {issue_data.get('state', '')}\n"
    
    # Team and assignee
    team_info = ""
    if issue_data.get('team'):
        team_name = issue_data['team'].get('name', '')
        team_key = issue_data['team'].get('key', '')
        team_info = f"{team_name} ({team_key})" if team_name else team_key
        context += f"Team: {team_info}\n"
    
    if issue_data.get('assignee') and issue_data['assignee'].get('name'):
        context += f"Assignee: {issue_data['assignee']['name']}\n"
    
    # Cycle
    if issue_data.get('cycle') and issue_data['cycle'].get('name'):
        context += f"Cycle: {issue_data['cycle']['name']}\n"
    
    # Priority
    if issue_data.get('priority'):
        context += f"Priority: {issue_data.get('priority')}\n"
    
    # Parent issue
    if issue_data.get('parent') and issue_data['parent'].get('id'):
        parent_title = issue_data['parent'].get('title', 'Unknown')
        parent_id = issue_data['parent'].get('id', '')
        context += f"Parent: {parent_title} (ID: {parent_id})\n"
    
    # Description
    description = issue_data.get('description', '')
    context += f"\nDescription:\n{description or 'No description provided.'}\n"
    
    # Comments
    comments = issue_data.get('comments', [])
    if comments:
        context += f"\nComments ({len(comments)}):\n"
        for comment in comments:
            user = comment.get('user', '')
            if isinstance(user, dict):
                user = user.get('name', 'Unknown')
            created_at = comment.get('created_at', 'Unknown date')
            body = comment.get('body', '')
            context += f"- {user} ({created_at}):\n  {body}\n\n"
    
    # Children
    children = issue_data.get('children', [])
    if children:
        context += f"\nSub-issues ({len(children)}):\n"
        for child in children:
            child_title = child.get('title', 'Unknown')
            child_id = child.get('id', '')
            context += f"- {child_title} (ID: {child_id})\n"
    
    # Dates
    context += f"\nCreated: {issue_data.get('created_at', '')}\n"
    if issue_data.get('updated_at'):
        context += f"Updated: {issue_data.get('updated_at')}\n"
    if issue_data.get('completed_at'):
        context += f"Completed: {issue_data.get('completed_at')}\n"
    
    return context

def import_linear_data(data_file: str):
    """
    Import data from a Linear JSON export into PostgreSQL using the simplified schema.
    
    Args:
        data_file: Path to the JSON file containing Linear data
    """
    logger.info(f"Importing data from {data_file}")
    
    # Load the data
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON data: {str(e)}")
        return
    
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Start transaction
                conn.autocommit = False
                
                # Import issues directly to the simplified schema
                logger.info("Importing issues to simplified schema...")
                issues_to_import = []
                
                for issue in data.get('issues', []):
                    # Get the actual issue data from the nested structure
                    issue_data = issue.get('data', {})
                    
                    # Skip issues with no ID
                    issue_id = issue_data.get('id', '')
                    if not issue_id:
                        continue
                    
                    # Fix empty team values
                    if 'team' not in issue_data or not issue_data['team']:
                        issue_data['team'] = {'id': '', 'key': '', 'name': ''}
                    
                    # Ensure cycle has a name
                    if 'cycle' in issue_data and issue_data['cycle'] and not issue_data['cycle'].get('name') and issue_data['cycle'].get('number'):
                        issue_data['cycle']['name'] = f"Cycle {issue_data['cycle']['number']}"
                    
                    # Generate full context
                    full_context = create_full_context(issue_data)
                    
                    # Store the issue with current timestamp
                    current_time = datetime.now().isoformat()
                    issues_to_import.append((
                        issue_id,
                        json.dumps(issue_data),
                        full_context,
                        issue_data.get('created_at', current_time),
                        issue_data.get('updated_at', current_time),
                    ))
                
                if issues_to_import:
                    execute_values(
                        cur,
                        """
                        INSERT INTO issues_simplified (id, data, full_context, created_at, updated_at)
                        VALUES %s
                        ON CONFLICT (id) DO UPDATE SET
                            data = EXCLUDED.data,
                            full_context = EXCLUDED.full_context,
                            created_at = EXCLUDED.created_at,
                            updated_at = EXCLUDED.updated_at
                        """,
                        issues_to_import
                    )
                    logger.info(f"Imported {len(issues_to_import)} issues into simplified schema")
                
                # Commit transaction
                conn.commit()
                logger.info("Data import completed successfully")
                
            except Exception as e:
                conn.rollback()
                logger.error(f"Error importing data: {str(e)}")
                raise

def check_existing_data():
    """Check if there is existing data in the database tables."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                # Check issues table
                cur.execute("SELECT COUNT(*) FROM issues_simplified")
                issues_count = cur.fetchone()[0]
                
                # Check embeddings table
                cur.execute("SELECT COUNT(*) FROM embeddings_simplified")
                embeddings_count = cur.fetchone()[0]
                
                return {
                    "issues_count": issues_count,
                    "embeddings_count": embeddings_count,
                    "has_data": issues_count > 0 or embeddings_count > 0
                }
            except Exception as e:
                logger.error(f"Error checking existing data: {str(e)}")
                return {"has_data": False, "issues_count": 0, "embeddings_count": 0}

def clear_existing_data():
    """Clear all data from the database tables."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            try:
                print("Clearing existing data from database...")
                
                # Use CASCADE to handle the foreign key constraints properly
                cur.execute("TRUNCATE TABLE issues_simplified, embeddings_simplified CASCADE")
                logger.info("Cleared all tables with CASCADE")
                print("✓ Cleared all tables successfully")
                
                conn.commit()
                return True
            except Exception as e:
                conn.rollback()
                logger.error(f"Error clearing data: {str(e)}")
                print(f"✗ Error clearing data: {str(e)}")
                return False

def main():
    """Import data from all Linear JSON files in the 1_month_data folder."""
    # Initialize the connection pool
    DatabasePool.initialize(minconn=1, maxconn=10)
    
    try:
        # Add command line arguments
        parser = argparse.ArgumentParser(description="Import Linear data from JSON files to PostgreSQL database")
        parser.add_argument('--clean', action='store_true', help='Automatically clear all existing data without prompting')
        args = parser.parse_args()
        
        folder_path = "1_month_data"
        
        # First, check for existing data
        existing_data = check_existing_data()
        if existing_data["has_data"]:
            print(f"\nFound existing data in database:")
            print(f"- Issues: {existing_data['issues_count']}")
            print(f"- Embeddings: {existing_data['embeddings_count']}")
            
            if args.clean:
                print("\nAutomatically clearing all existing data (--clean flag was used)")
                success = clear_existing_data()
                if not success:
                    print("Import canceled due to error clearing data.")
                    return
            else:
                user_input = input("\nWould you like to clear all existing data before import? (y/n): ")
                if user_input.lower() == 'y':
                    success = clear_existing_data()
                    if not success:
                        print("Import canceled due to error clearing data.")
                        return
                else:
                    print("\nContinuing import without clearing data. New issues will overwrite existing ones with the same ID.")
        
        total_issues_imported = 0
        
        # List files in order of week number to ensure consistent processing
        week_files = ["week_1.json", "week_2.json", "week_3.json", "week_4.json"]
        
        print(f"\nImporting Linear data from all week files in {folder_path}:")
        print("=" * 60)
        
        for file in week_files:
            file_path = os.path.join(folder_path, file)
            if not os.path.exists(file_path):
                print(f"✗ File not found: {file_path}")
                continue
                
            try:
                # Count issues before import
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    issues_count = len(data.get('issues', []))
                
                print(f"\nProcessing {file} ({issues_count} issues)...")
                import_linear_data(file_path)
                
                total_issues_imported += issues_count
                print(f"✓ Successfully imported {issues_count} issues from {file}")
            except Exception as e:
                logger.error(f"Error importing data from {file}: {str(e)}")
                print(f"✗ Error importing data from {file}: {str(e)}")
        
        print("\n" + "=" * 60)
        print(f"Import completed: {total_issues_imported} total issues imported from {len(week_files)} files")
        
        # Verify data in database
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # Count issues
                    cur.execute("SELECT COUNT(*) FROM issues_simplified")
                    db_issue_count = cur.fetchone()[0]
                    
                    # Count by cycle
                    cur.execute("SELECT data->'cycle'->>'name' as cycle_name, COUNT(*) FROM issues_simplified GROUP BY cycle_name ORDER BY cycle_name")
                    cycle_counts = cur.fetchall()
                    
                    print(f"\nDatabase summary:")
                    print(f"- Total issues in database: {db_issue_count}")
                    print("- Issues per cycle:")
                    for cycle_name, count in cycle_counts:
                        print(f"  • {cycle_name}: {count} issues")
                    
                except Exception as e:
                    print(f"Error checking database: {str(e)}")
        
        print("\nNext step: Run linear_rag_embeddings.py to generate embeddings for the imported data")
    
    finally:
        # Clean up the connection pool
        DatabasePool.close_all()

if __name__ == "__main__":
    main() 