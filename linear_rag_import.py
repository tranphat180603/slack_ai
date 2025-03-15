"""
Import data from Linear JSON exports into the PostgreSQL database.
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
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    
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

def import_linear_data(data_file: str):
    """
    Import data from a Linear JSON export into PostgreSQL.
    
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
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Start transaction
        conn.autocommit = False
        
        # Import teams
        logger.info("Importing teams...")
        teams = []
        for team in data.get('teams', []):
            teams.append((
                team.get('id', ''),
                team.get('name', ''),
                team.get('key', ''),
                team.get('description', '')
            ))
        
        if teams:
            execute_values(
                cur,
                """
                INSERT INTO teams (id, name, key, description)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    key = EXCLUDED.key,
                    description = EXCLUDED.description
                """,
                teams
            )
            logger.info(f"Imported {len(teams)} teams")
        
        # Import cycles
        logger.info("Importing cycles...")
        cycles = []
        for cycle in data.get('cycles', []):
            team_id = None
            if cycle.get('team_key'):
                # Look up team ID from key
                team_key = cycle.get('team_key')
                cur.execute("SELECT id FROM teams WHERE key = %s", (team_key,))
                result = cur.fetchone()
                if result:
                    team_id = result[0]
            
            cycles.append((
                cycle.get('id', ''),
                cycle.get('name', ''),
                cycle.get('number'),
                team_id,
                cycle.get('starts_at'),
                cycle.get('ends_at'),
                cycle.get('progress', 0)
            ))
        
        if cycles:
            execute_values(
                cur,
                """
                INSERT INTO cycles (id, name, number, team_id, starts_at, ends_at, progress)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    number = EXCLUDED.number,
                    team_id = EXCLUDED.team_id,
                    starts_at = EXCLUDED.starts_at,
                    ends_at = EXCLUDED.ends_at,
                    progress = EXCLUDED.progress
                """,
                cycles
            )
            logger.info(f"Imported {len(cycles)} cycles")
        
        # Import employees (extracted from issues)
        logger.info("Importing employees...")
        # Use dictionary to properly deduplicate by employee ID
        employee_dict = {}
        for issue in data.get('issues', []):
            if issue.get('assignee') and issue['assignee'].get('id'):
                emp_id = issue['assignee'].get('id', '')
                # Skip if empty ID
                if not emp_id:
                    continue
                    
                emp_name = issue['assignee'].get('name', 'Unknown User')
                team_id = None
                if issue.get('team') and issue['team'].get('id'):
                    team_id = issue['team']['id']
                
                # Only add employee or update team_id if needed
                if emp_id not in employee_dict:
                    employee_dict[emp_id] = {
                        'id': emp_id,
                        'name': emp_name,
                        'email': None,  # email not available
                        'team_id': team_id
                    }
                # If we already have this employee but team_id is None, update it
                elif employee_dict[emp_id]['team_id'] is None and team_id is not None:
                    employee_dict[emp_id]['team_id'] = team_id
        
        # Convert dictionary to list of tuples for database insertion
        employees = [(
            emp['id'],
            emp['name'],
            emp['email'],
            emp['team_id']
        ) for emp in employee_dict.values()]
        
        if employees:
            execute_values(
                cur,
                """
                INSERT INTO employees (id, name, email, team_id)
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    team_id = COALESCE(EXCLUDED.team_id, employees.team_id)
                """,
                employees
            )
            logger.info(f"Imported {len(employees)} employees")
        
        # Import issues using a two-phase approach to handle parent-child relationships
        logger.info("Importing issues (phase 1: initial import with NULL parent_id)...")
        
        # Prepare all issues, but temporarily set parent_id to NULL
        issues_phase1 = []
        parent_child_mapping = {}  # Store the original parent-child relationships
        
        for issue in data.get('issues', []):
            # Skip issues with no ID
            issue_id = issue.get('id', '')
            if not issue_id:
                continue
                
            # Determine cycle ID
            cycle_id = None
            if 'cycle' in issue and issue['cycle']:
                cycle_id = issue['cycle'].get('id')
                # Convert empty string to None
                if cycle_id == '':
                    cycle_id = None
            
            # Save original parent_id but set to NULL for initial import
            parent_id = issue.get('parent', {}).get('id') if issue.get('parent') else None
            if parent_id == '':
                parent_id = None
            
            # Store the original parent-child relationship if parent_id exists
            if parent_id:
                parent_child_mapping[issue_id] = parent_id
                
            # Handle assignee properly
            assignee_id = issue.get('assignee', {}).get('id') if issue.get('assignee') else None
            if assignee_id == '':
                assignee_id = None
                
            # Handle team properly
            team_id = issue.get('team', {}).get('id') if issue.get('team') else None
            if team_id == '':
                team_id = None
            
            issues_phase1.append((
                issue_id,
                issue.get('title', ''),
                issue.get('description', ''),
                issue.get('state', ''),
                issue.get('state_type', ''),
                cycle_id,
                assignee_id,
                team_id,
                issue.get('priority'),
                issue.get('estimate'),
                issue.get('created_at'),
                issue.get('updated_at'),
                issue.get('completed_at'),
                None  # Set parent_id to NULL for initial import
            ))
        
        if issues_phase1:
            execute_values(
                cur,
                """
                INSERT INTO issues (
                    id, title, description, state, state_type, cycle_id, assignee_id, 
                    team_id, priority, estimate, created_at, updated_at, completed_at, parent_id
                )
                VALUES %s
                ON CONFLICT (id) DO UPDATE SET
                    title = EXCLUDED.title,
                    description = EXCLUDED.description,
                    state = EXCLUDED.state,
                    state_type = EXCLUDED.state_type,
                    cycle_id = EXCLUDED.cycle_id,
                    assignee_id = EXCLUDED.assignee_id,
                    team_id = EXCLUDED.team_id,
                    priority = EXCLUDED.priority,
                    estimate = EXCLUDED.estimate,
                    updated_at = EXCLUDED.updated_at,
                    completed_at = EXCLUDED.completed_at
                    -- Do not update parent_id yet
                """,
                issues_phase1
            )
            logger.info(f"Imported {len(issues_phase1)} issues (phase 1)")
            
            # Phase 2: Update parent_id values now that all issues exist in the database
            if parent_child_mapping:
                logger.info(f"Importing issues (phase 2: updating parent-child relationships)...")
                updated_count = 0
                
                # Use a batched approach for updating parent_id values
                batch_size = 100
                parent_child_items = list(parent_child_mapping.items())
                for i in range(0, len(parent_child_items), batch_size):
                    batch = parent_child_items[i:i+batch_size]
                    update_data = []
                    
                    for child_id, parent_id in batch:
                        # Verify the parent exists
                        cur.execute("SELECT 1 FROM issues WHERE id = %s", (parent_id,))
                        if cur.fetchone():
                            update_data.append((parent_id, child_id))
                        else:
                            logger.warning(f"Parent issue {parent_id} not found for child {child_id}. Skipping relationship.")
                    
                    if update_data:
                        # Use execute_values for efficient batch updates
                        execute_values(
                            cur,
                            """
                            UPDATE issues
                            SET parent_id = data.parent_id
                            FROM (VALUES %s) AS data(parent_id, child_id)
                            WHERE issues.id = data.child_id
                            """,
                            update_data,
                            template="(%s, %s)"
                        )
                        updated_count += len(update_data)
                
                logger.info(f"Updated {updated_count} parent-child relationships (phase 2)")
        
        # Import comments
        logger.info("Importing comments...")
        comments = []
        comment_id = 1  # For generating IDs if needed
        
        for issue in data.get('issues', []):
            issue_id = issue.get('id')
            # Skip if issue ID is missing or empty
            if not issue_id:
                continue
                
            for comment in issue.get('comments', []):
                comment_id_value = comment.get('id')
                # Generate an ID if empty or missing
                if not comment_id_value or comment_id_value == '':
                    comment_id_value = f"gen_comment_{comment_id}"
                    comment_id += 1
                
                # Try to get user ID if user info is available
                user_id = None
                if 'user' in comment and isinstance(comment['user'], dict) and 'id' in comment['user']:
                    user_id = comment['user']['id']
                    # Skip empty user IDs
                    if user_id == '':
                        user_id = None
                
                body = comment.get('body', '')
                created_at = comment.get('created_at')
                
                comments.append((
                    comment_id_value,
                    issue_id,
                    user_id,
                    body,
                    created_at
                ))
        
        if comments:
            execute_values(
                cur,
                """
                INSERT INTO comments (id, issue_id, user_id, body, created_at)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                comments
            )
            logger.info(f"Imported {len(comments)} comments")
        
        # Commit transaction
        conn.commit()
        logger.info("Data import completed successfully")
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error importing data: {str(e)}")
        raise
    finally:
        cur.close()
        conn.close()

def main():
    folder_path = "1_month_data"
    for file in os.listdir(folder_path):
        try:
            if file.endswith(".json"):
                file_path = os.path.join(folder_path, file)
                import_linear_data(file_path)
                print(f"✓ Successfully imported Linear data from {file_path}")
        except Exception as e:
            logger.error(f"Error importing data from {file}: {str(e)}")
            print(f"✗ Error importing data from {file}: {str(e)}")

if __name__ == "__main__":
    main() 