"""
Script to synchronize Linear data with the semantic search database.
This script fetches Linear issues, projects, and comments and generates embeddings for them.
"""

import os
import logging
import argparse
import time
import json
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import traceback
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_rag_sync")

# Add parent directory to path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules
from ops_linear_db.linear_client import LinearClient, LinearError
from ops_linear_db.linear_rag_embeddings import (
    store_issue_embedding,
    store_project_embedding,
    store_comment_embedding,
    clear_embeddings
)
from ops_linear_db.db_pool import get_db_connection

load_dotenv()

# Disable GQL transport request logs
logging.getLogger('gql.transport.requests').setLevel(logging.WARNING)

# Disable HTTPX logs (OpenAI API requests)
logging.getLogger('httpx').setLevel(logging.WARNING)

def get_teams() -> List[str]:
    """"
    return an enumerate of the teams in the LINEAR_DEFAULT_TEAMS list
    """
    return ["OPS", "ENG", "AI", "RES", "PRO", "MKT"]

def get_last_cycles(linear_client: LinearClient, team_key: str, cycle_count: int = 4) -> List[Dict[str, Any]]:
    """
    Get the last N cycles for a team.
    
    Args:
        linear_client: Initialized Linear client
        team_key: Team key
        cycle_count: Number of recent cycles to fetch
        
    Returns:
        List of cycle data
    """
    try:
        all_cycles = linear_client.getAllCycles(team_key)
        
        # Sort cycles by start date (descending)
        sorted_cycles = sorted(
            all_cycles, 
            key=lambda c: c.get("startsAt", "2000-01-01"), 
            reverse=True
        )
        
        # Take the most recent cycles
        recent_cycles = sorted_cycles[:cycle_count]
        
        logger.info(f"Found {len(recent_cycles)} recent cycles for team {team_key}")
        for cycle in recent_cycles:
            logger.info(f"  - Cycle: {cycle.get('number')} - {cycle.get('startsAt')} to {cycle.get('endsAt')}")
            
        return recent_cycles
    
    except LinearError as e:
        logger.error(f"Error getting cycles for team {team_key}: {str(e)}")
        return []

def get_issues_for_cycles(linear_client: LinearClient, team_key: str, cycle_ids: List[str]) -> List[Dict[str, Any]]:
    """
    Get all issues for specified cycles.
    
    Args:
        linear_client: Initialized Linear client
        team_key: Team key
        cycle_ids: List of cycle IDs
        
    Returns:
        List of issue data
    """
    all_issues = []
    
    for cycle_id in cycle_ids:
        try:
            # Filter issues by team and cycle
            filter_criteria = {
                "team": {"key": {"eq": team_key}},
                "cycle": {"id": {"eq": cycle_id}}
            }
            cycle_issues = linear_client.filterIssues(filter_criteria)
            logger.info(f"Found {len(cycle_issues)} issues for cycle {cycle_id}")
            
            all_issues.extend(cycle_issues)
            
        except LinearError as e:
            logger.error(f"Error getting issues for cycle {cycle_id}: {str(e)}")
    
    return all_issues

def get_active_projects(linear_client: LinearClient, team_key: str) -> List[Dict[str, Any]]:
    """
    Get all active projects for a team.
    
    Args:
        linear_client: Initialized Linear client
        team_key: Team key
        
    Returns:
        List of project data
    """
    try:
        # Filter for active projects (not completed or canceled)
        # Note: The "team" filter is not available in ProjectFilter, so we'll filter in Python
        filter_criteria = {
            "state": {"neq": "completed"}
        }
        
        # Get all projects and filter by team in Python
        projects = linear_client.filterProjects(filter_criteria)
        
        # Filter projects that belong to the specified team
        team_projects = [
            project for project in projects
            if project.get("team", {}).get("key") == team_key
        ]
        
        logger.info(f"Found {len(team_projects)} active projects for team {team_key}")
        
        return team_projects
    
    except LinearError as e:
        logger.error(f"Error getting projects for team {team_key}: {str(e)}")
        return []

def get_comments_for_issues(linear_client: LinearClient, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all comments for specified issues.
    
    Args:
        linear_client: Initialized Linear client
        issues: List of issue objects
        
    Returns:
        List of comment objects
    """
    all_comments = []
    
    # Limit to 50 issues to avoid API rate limits
    for issue in issues:
        try:
            # Use the actual issue ID (UUID) from the issue object, not the issue number
            issue_id = issue.get("id")
            if not issue_id:
                continue
                
            # Filter criteria: get comments for this issue where body is not empty
            filter_criteria = {
                "issue": {"id": {"eq": issue_id}},
                "body": {"neq": ""}  # Filter out empty comment bodies
            }
            
            comments = linear_client.filterComments(filter_criteria)
            
            if comments:
                # Attach the issue ID to each comment for reference
                for comment in comments:
                    if "issue" not in comment:
                        comment["issue"] = {"id": issue_id, "number": issue.get("number")}
                
                all_comments.extend(comments)
            
        except LinearError as e:
            logger.error(f"Error getting comments for issue {issue.get('number')}: {str(e)}")
    
    logger.info(f"Found {len(all_comments)} comments for {len(issues)} issues")
    return all_comments

def clean_old_embeddings(database_conn, team_key: str, keep_cycle_ids: List[str]):
    """
    Remove embeddings for issues from old cycles.
    
    Args:
        database_conn: Database connection
        team_key: Team key
        keep_cycle_ids: List of cycle IDs to keep
    """
    try:
        with database_conn.cursor() as cur:
            # Convert cycle IDs to JSON array format for Postgres
            cycle_ids_json = json.dumps(keep_cycle_ids)
            
            # Delete issues not in the specified cycles
            cur.execute("""
            DELETE FROM linear_embeddings
            WHERE 
                object_type = 'Issue' 
                AND metadata->>'team_key' = %s
                AND (
                    metadata->>'cycle_id' IS NULL 
                    OR metadata->>'cycle_id' NOT IN (SELECT jsonb_array_elements_text(%s::jsonb))
                )
            """, (team_key, cycle_ids_json))
            
            deleted_count = cur.rowcount
            database_conn.commit()
            
            logger.info(f"Removed {deleted_count} embeddings for old issues from team {team_key}")
            
    except Exception as e:
        logger.error(f"Error cleaning old embeddings: {str(e)}")
        database_conn.rollback()

def sync_team_data(linear_client: LinearClient, team_key: str, cycle_count: int = 4):
    """
    Sync data for a specific team, focusing on recent cycles.
    
    Args:
        linear_client: Initialized Linear client
        team_key: Team key
        cycle_count: Number of recent cycles to include
    """
    logger.info(f"Syncing data for team {team_key}")
    
    # Get recent cycles
    recent_cycles = get_last_cycles(linear_client, team_key, cycle_count)
    cycle_ids = [cycle.get("id") for cycle in recent_cycles if cycle.get("id")]
    
    if not cycle_ids:
        logger.warning(f"No recent cycles found for team {team_key}")
        return
    
    # Get issues for these cycles
    issues = get_issues_for_cycles(linear_client, team_key, cycle_ids)
    
    logger.info(f"Processing {len(issues)} issues from {len(cycle_ids)} cycles")
    
    # Get active projects
    projects = get_active_projects(linear_client, team_key)
    
    # Get comments for issues - pass full issue objects instead of just IDs
    comments = get_comments_for_issues(linear_client, issues)
    
    # Process and store embeddings
    processed_count = {
        "issues": 0,
        "projects": 0,
        "comments": 0
    }
    
    # Store issue embeddings
    for issue in issues:
        try:
            # Skip if issue is None or not a dictionary
            if not issue or not isinstance(issue, dict):
                logger.warning(f"Skipping invalid issue object: {type(issue)}")
                continue
                
            # Get the issue ID for logging
            issue_id = issue.get('id', 'unknown')
            issue_number = issue.get('number', 'unknown')
            
            logger.info(f"Processing issue #{issue_number} ({issue_id})")
            
            if store_issue_embedding(issue):
                processed_count["issues"] += 1
            else:
                logger.warning(f"Failed to store embedding for issue #{issue_number}")
        except Exception as e:
            logger.error(f"Error processing issue {issue.get('id', 'unknown') if isinstance(issue, dict) else 'unknown'}: {str(e)}")
            continue
    
    # Store project embeddings
    for project in projects:
        try:
            # Skip if project is None or not a dictionary
            if not project or not isinstance(project, dict):
                logger.warning(f"Skipping invalid project object: {type(project)}")
                continue
                
            # Get the project ID for logging
            project_id = project.get('id', 'unknown')
            project_name = project.get('name', 'unknown')
            
            logger.info(f"Processing project '{project_name}' ({project_id})")
            
            if store_project_embedding(project):
                processed_count["projects"] += 1
            else:
                logger.warning(f"Failed to store embedding for project '{project_name}'")
        except Exception as e:
            logger.error(f"Error processing project {project.get('id', 'unknown') if isinstance(project, dict) else 'unknown'}: {str(e)}")
            continue
    
    # Store comment embeddings
    for comment in comments:
        try:
            # Skip if comment is None or not a dictionary
            if not comment or not isinstance(comment, dict):
                logger.warning(f"Skipping invalid comment object: {type(comment)}")
                continue
                
            # Get the comment ID for logging
            comment_id = comment.get('id', 'unknown')
                
            # Find the parent issue
            issue_obj = comment.get("issue", {})
            if not isinstance(issue_obj, dict):
                logger.warning(f"Skipping comment {comment_id}: Invalid issue reference")
                continue
                
            issue_id = issue_obj.get("id")
            if not issue_id:
                logger.warning(f"Skipping comment {comment_id}: Missing issue ID")
                continue
                
            parent_issue = next((i for i in issues if i.get("id") == issue_id), None)
            
            # Skip comments with no content
            comment_body = comment.get("body", "").strip()
            if not comment_body or len(comment_body) < 10:  # Skip empty or very short comments
                logger.info(f"Skipping comment {comment_id}: Content too short or empty")
                continue
                
            # Only store the comment if we have a valid parent issue
            if parent_issue is not None and isinstance(parent_issue, dict):
                if store_comment_embedding(comment, parent_issue):
                    processed_count["comments"] += 1
            else:
                logger.warning(f"Skipping comment {comment_id}: No matching parent issue found")
        except Exception as e:
            logger.error(f"Error processing comment {comment.get('id', 'unknown')}: {str(e)}")
            continue
    
    logger.info(f"Sync completed for team {team_key}:")
    logger.info(f"  - Issues: {processed_count['issues']}/{len(issues)}")
    logger.info(f"  - Projects: {processed_count['projects']}/{len(projects)}")
    logger.info(f"  - Comments: {processed_count['comments']}/{len(comments)}")
    
    # Clean up old embeddings
    with get_db_connection() as conn:
        clean_old_embeddings(conn, team_key, cycle_ids)

def record_sync_timestamp():
    """Record the timestamp of the last successful sync."""
    try:
        timestamp = datetime.now().isoformat()
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create a simple metadata table if it doesn't exist
                cur.execute("""
                CREATE TABLE IF NOT EXISTS linear_rag_metadata (
                    key VARCHAR(50) PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Store last sync timestamp
                timestamp_data = json.dumps({"timestamp": timestamp})
                cur.execute("""
                INSERT INTO linear_rag_metadata (key, value)
                VALUES ('last_sync', %s::jsonb)
                ON CONFLICT (key) DO UPDATE
                SET value = %s::jsonb, updated_at = CURRENT_TIMESTAMP
                """, (timestamp_data, timestamp_data))
                
                conn.commit()
                
        logger.info(f"Recorded sync timestamp: {timestamp}")
    except Exception as e:
        logger.error(f"Error recording sync timestamp: {str(e)}")

def check_last_sync_time():
    """Check when the last sync occurred."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT value FROM linear_rag_metadata WHERE key = 'last_sync'")
                result = cur.fetchone()
                
                if result:
                    # Safely parse the JSON string
                    if isinstance(result[0], str):
                        last_sync_data = json.loads(result[0])
                    else:
                        # If it's already a dict, use it directly
                        last_sync_data = result[0]
                    
                    last_sync = last_sync_data.get("timestamp")
                    if last_sync:
                        last_sync_time = datetime.fromisoformat(last_sync)
                        hours_since_sync = (datetime.now() - last_sync_time).total_seconds() / 3600
                        
                        return hours_since_sync
        
        # If we reach here, either no sync record was found or it couldn't be parsed
        return None
    except Exception as e:
        logger.warning(f"Could not check last sync time: {str(e)}")
        return None

def main():
    """Run the sync process with command line arguments."""
    parser = argparse.ArgumentParser(description="Linear RAG Daily Sync Tool")
    
    # Basic options
    parser.add_argument('--teams', nargs='+', type=str,
                      help='Team keys to sync (default: all teams)')
    parser.add_argument('--cycles', type=int, default=4,
                      help='Number of recent cycles to include (default: 4)')
    parser.add_argument('--force', action='store_true',
                      help='Force full sync even if recently synced')
                      
    args = parser.parse_args()
    
    # Initialize Linear client
    linear_api_key = os.environ.get('LINEAR_API_KEY')
    if not linear_api_key:
        logger.error("ERROR: LINEAR_API_KEY environment variable not set")
        return
    
    linear = LinearClient(linear_api_key)
    
    # Check last sync time
    if not args.force:
        hours_since_sync = check_last_sync_time()
        
        if hours_since_sync is not None and hours_since_sync < 20:  # Less than 20 hours ago
            logger.info(f"Last sync was {hours_since_sync:.1f} hours ago. Skipping (use --force to override)")
            return
    
    # Get teams to process
    if args.teams:
        team_keys = args.teams
        logger.info(f"Processing specified teams: {', '.join(team_keys)}")
    else:
        # Fetch all teams
        teams = get_teams()
        team_keys = teams
        logger.info(f"Processing all teams: {', '.join(team_keys)}")
    
    # Process each team
    for team_key in team_keys:
        try:
            sync_team_data(linear, team_key, args.cycles)
        except Exception as e:
            logger.error(f"Error processing team {team_key}: {str(e)}")
    
    # Record successful sync
    record_sync_timestamp()
    
    logger.info("Sync process completed successfully")

def check_database_record():
    """Check if the database record exists."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM linear_embeddings LIMIT 10")
                result = cur.fetchall()
                logger.info(f"Sample database record: {result}")
    except Exception as e:
        logger.error(f"Error checking database record: {str(e)}")
                

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting sync at {start_time.isoformat()}")
    
    try:
        main()
    except Exception as e:
        logger.error(f"Sync process failed: {str(e)}")
        
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    logger.info(f"Sync completed at {end_time.isoformat()} (Duration: {duration:.1f} seconds)") 

    check_database_record()