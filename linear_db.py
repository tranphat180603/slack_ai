#!/usr/bin/env python3
"""
Linear RAG Database Pipeline

This script handles both daily and weekly updates for the Linear RAG system:
Daily tasks:
- Fetches the most recent 4 cycles from Linear API
- Updates the data fields in database tables

Weekly tasks (in addition to daily tasks):
- Clears the existing database
- Generates new embeddings for all data

The script runs continuously using the schedule library to maintain the update schedule.
"""
import os
import sys
import logging
import argparse
import json
import pathlib
import time
import schedule
from datetime import datetime
from dotenv import load_dotenv

# Import functionality from existing modules
# We use import statements rather than subprocess to maintain proper error handling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from linear_client import LinearClient
import linear_data_gen
import linear_rag_db_import
import linear_rag_embeddings
import linear_rag_db_create
# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_db")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Linear RAG Database Scheduled Pipeline")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--run-now', action='store_true', help='Run the pipeline immediately before starting schedule')
    parser.add_argument('--weekly', action='store_true', help='Also run weekly update when using --run-now')
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment based on arguments."""
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)

def run_linear_data_generation():
    """Run the data generation process to fetch and process Linear data."""
    logger.info("Starting Linear data generation process...")
    
    try:
        # Instead of importing and running the main() function, we'll implement 
        # the core functionality here to have more control
        api_key = os.environ.get("LINEAR_API_KEY")
        if not api_key:
            logger.error("ERROR: Please set the LINEAR_API_KEY environment variable")
            return False
        
        # Initialize the Linear client
        client = LinearClient(api_key)
        logger.info("Linear client initialized")
        
        # Step 1: Get all teams
        teams = client.get_all_teams()
        if not teams:
            logger.error("No teams found in Linear workspace")
            return False
        
        # Filter out teams we don't want to include
        teams_to_remove = ["MGT", "DAO2"]
        teams = [team for team in teams if team.key not in teams_to_remove]
        
        logger.info(f"Found {len(teams)} teams in workspace")
        
        # Create a directory for storing data
        data_dir = pathlib.Path("cycles_data")
        data_dir.mkdir(exist_ok=True)
        
        # Get reference cycle names from Research team
        research_cycles = linear_data_gen.get_research_team_cycles(client)
        if not research_cycles:
            logger.warning("Could not get cycle names from Research team. Using default naming.")
            return False
        
        logger.info(f"Using reference cycle names: Active={research_cycles['active']}, Recent={research_cycles['recent']}")
        
        # Step 2: Collect all cycles from all teams
        logger.info("Fetching last 4 cycles for all teams...")
        all_cycles = []
        
        for team in teams:
            logger.info(f"Checking team: {team.name} ({team.key})")
            
            # Get the recent cycles for this team (active + last 3 completed)
            team_cycles_data = client.get_team_recent_cycles(team_key=team.key, limit=4)
            
            if "error" not in team_cycles_data:
                # Add active cycle if exists
                active_cycle = team_cycles_data.get('active_cycle')
                if active_cycle:
                    # Use the Research team's active cycle name
                    original_name = active_cycle.get('name', '')
                    active_cycle['name'] = research_cycles['active']
                    
                    # Extract cycle number from name
                    cycle_number = linear_data_gen.extract_cycle_number(research_cycles['active'])
                    all_cycles.append({
                        'id': active_cycle.get('id'),
                        'team_key': team.key,
                        'team_name': team.name,
                        'name': research_cycles['active'],
                        'original_name': original_name,
                        'number': cycle_number,
                        'starts_at': active_cycle.get('startsAt', ''),
                        'ends_at': active_cycle.get('endsAt', ''),
                        'is_active': True
                    })
                    logger.info(f"  ✓ Active cycle found: {original_name} → {research_cycles['active']}")
                
                # Add recent cycles
                recent_cycles = team_cycles_data.get('recent_cycles', [])
                for i, cycle in enumerate(recent_cycles):
                    if i < len(research_cycles['recent']):
                        # Use the Research team's cycle name for this position
                        original_name = cycle.get('name', '')
                        cycle['name'] = research_cycles['recent'][i]
                        
                        # Extract cycle number from name
                        cycle_number = linear_data_gen.extract_cycle_number(research_cycles['recent'][i])
                        all_cycles.append({
                            'id': cycle.get('id'),
                            'team_key': team.key,
                            'team_name': team.name,
                            'name': research_cycles['recent'][i],
                            'original_name': original_name,
                            'number': cycle_number,
                            'starts_at': cycle.get('startsAt', ''),
                            'ends_at': cycle.get('endsAt', ''),
                            'is_active': False
                        })
                        logger.info(f"  ✓ Recent cycle {i+1}: {original_name} → {research_cycles['recent'][i]}")
                
                logger.info(f"  ✓ Retrieved {len(recent_cycles)} historical cycles")
            else:
                error_msg = team_cycles_data.get("error", "Unknown error")
                logger.error(f"  ✗ Error getting cycles for {team.key}: {error_msg}")
        
        logger.info(f"Collected {len(all_cycles)} cycles from all teams")
        
        # Step 3: Group cycles by standardized names (not by date)
        cycle_groups = linear_data_gen.group_cycles_by_name(all_cycles, research_cycles)
        logger.info(f"Grouped into {len(cycle_groups)} distinct cycle periods")
        
        # Step 4: Ensure we have correct week labeling (active cycle is week_4)
        # Sort groups by cycle number, highest first
        cycle_groups.sort(key=lambda g: g['cycle_number'], reverse=True)
        
        # Assign week numbers, ensuring active cycle is week_4
        for i, group in enumerate(cycle_groups[:4]):
            week_number = 4 - i  # Active cycle (highest number) is week_4, count down from there
            group['week_label'] = f"week_{week_number}"
            logger.info(f"Week {week_number}: {group['standardized_name']} ({group['starts_at']} to {group['ends_at']})")
        
        # Step 5: Process only the 4 most recent cycle groups
        cycles_to_process = cycle_groups[:4]
        for week_group in cycles_to_process:
            linear_data_gen.process_cycle_group(client, week_group, data_dir)
        
        logger.info("Summary of data collection:")
        for group in cycles_to_process:
            logger.info(f"  - {group['week_label']}: {group['standardized_name']} - {len(group.get('processed_issues', []))} issues")
        
        logger.info(f"Data has been saved to: {data_dir}")
        
        # Step 6: Remove any older JSON files to maintain the 4-cycle window
        for week_file in data_dir.glob("*.json"):
            filename = week_file.name
            # Keep only the week_1 through week_4 files and their summaries
            if not (filename.startswith("week_1") or 
                    filename.startswith("week_2") or 
                    filename.startswith("week_3") or 
                    filename.startswith("week_4")):
                logger.info(f"Removing old cycle file: {filename}")
                week_file.unlink()
        
        return True
    except Exception as e:
        logger.error(f"Error in Linear data generation: {str(e)}", exc_info=True)
        return False

def clear_database_tables():
    """Clear issues table and non-embedding data from embeddings table."""
    logger.info("Clearing existing data from database tables...")
    try:
        with linear_rag_db_create.get_db_connection() as conn:
            with conn.cursor() as cur:
                # First, clear the foreign key references in embeddings_simplified
                cur.execute("""
                    UPDATE embeddings_simplified 
                    SET 
                        content = NULL,
                        data = NULL,
                        issue_id = NULL
                """)
                
                # Then truncate the issues table
                cur.execute("TRUNCATE TABLE issues_simplified CASCADE")
                
                conn.commit()
                logger.info("Successfully cleared tables while preserving embeddings")
                return True
    except Exception as e:
        logger.error(f"Error clearing tables: {str(e)}")
        return False

def clear_embeddings_only():
    """Clear only the embeddings column."""
    logger.info("Clearing existing embeddings...")
    try:
        with linear_rag_db_create.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Only clear the embedding column
                cur.execute("""
                    UPDATE embeddings_simplified 
                    SET embedding = NULL
                """)
                
                conn.commit()
                logger.info("Successfully cleared embeddings column")
                return True
    except Exception as e:
        logger.error(f"Error clearing embeddings: {str(e)}")
        return False

def import_data_to_database():
    """Import the JSON data files into the database."""
    logger.info("Importing Linear data to database...")
    
    try:
        data_dir = pathlib.Path("cycles_data")
        
        # Get the list of week files to import (week_1.json through week_4.json)
        week_files = [
            data_dir / "week_1.json",
            data_dir / "week_2.json",
            data_dir / "week_3.json",
            data_dir / "week_4.json"
        ]
        
        # Check if files exist
        existing_files = [f for f in week_files if f.exists()]
        if not existing_files:
            logger.error("No week files found to import")
            return False
        
        logger.info(f"Found {len(existing_files)} week files to import")
        
        # Import each file
        total_issues = 0
        for week_file in existing_files:
            try:
                # Count issues in the file for reporting
                with open(week_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    issue_count = len(data.get('issues', []))
                
                logger.info(f"Importing {week_file.name} with {issue_count} issues...")
                linear_rag_db_import.import_linear_data(str(week_file))
                total_issues += issue_count
            except Exception as e:
                logger.error(f"Error importing {week_file}: {str(e)}")
        
        # Verify database state after import
        db_stats = linear_rag_db_import.check_existing_data()
        logger.info(f"Database now has {db_stats['issues_count']} issues")
        
        return total_issues > 0
    except Exception as e:
        logger.error(f"Error in data import: {str(e)}", exc_info=True)
        return False

def generate_embeddings():
    """Generate embeddings for the imported data."""
    logger.info("Generating embeddings for Linear data...")
    try:
        linear_rag_embeddings.generate_embeddings_for_issues()
        return True
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        return False

def update_full_context_in_database():
    """Update the full_context field for all issues to only include title and description."""
    logger.info("Updating full_context for all issues...")
    try:
        with linear_rag_db_create.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Update full_context for all issues
                cur.execute("""
                    UPDATE issues_simplified
                    SET full_context = 
                        'Title: ' || (data->>'title') || E'\n\nDescription:\n' || COALESCE(data->>'description', '')
                """)
                
                conn.commit()
                
                # Get count of updated records
                cur.execute("SELECT COUNT(*) FROM issues_simplified")
                count = cur.fetchone()[0]
                logger.info(f"Updated full_context for {count} issues")
                
                return True
    except Exception as e:
        logger.error(f"Error updating full_context: {str(e)}")
        return False

def update_embeddings_data():
    """Update the data field in embeddings_simplified table to match issues_simplified."""
    logger.info("Updating data field in embeddings_simplified table...")
    try:
        with linear_rag_db_create.get_db_connection() as conn:
            with conn.cursor() as cur:
                # First, restore issue_id relationships by matching Linear IDs
                cur.execute("""
                    UPDATE embeddings_simplified e
                    SET 
                        issue_id = i.id,
                        data = i.data,
                        content = CONCAT(
                            'Title: ',
                            i.data->>'title',
                            E'\n\nDescription:\n',
                            COALESCE(i.data->>'description', '')
                        )
                    FROM issues_simplified i
                    WHERE (e.data->>'id')::text = (i.data->>'id')::text
                """)
                
                # Log how many rows were updated
                cur.execute("SELECT COUNT(*) FROM embeddings_simplified WHERE issue_id IS NOT NULL")
                count = cur.fetchone()[0]
                logger.info(f"Updated {count} rows in embeddings_simplified")
                
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error updating embeddings data: {str(e)}")
        return False

def update_issue_references():
    """Update parent and children issue IDs to use Linear format (e.g., MKT-1888)."""
    logger.info("Updating parent and children issue references...")
    try:
        with linear_rag_db_create.get_db_connection() as conn:
            with conn.cursor() as cur:
                # First, create a mapping of UUID to Linear ID
                cur.execute("""
                    SELECT id, data->>'id' as linear_id
                    FROM issues_simplified
                    WHERE data->>'id' LIKE '%-%'
                """)
                id_mapping = {row[0]: row[1] for row in cur.fetchall()}
                
                # Update parent references
                cur.execute("""
                    UPDATE issues_simplified
                    SET data = jsonb_set(
                        data,
                        '{parent}',
                        jsonb_build_object(
                            'id',
                            COALESCE(
                                (SELECT data->>'id'
                                FROM issues_simplified parent
                                WHERE parent.id = (data->'parent'->>'id')::text
                                LIMIT 1),
                                data->'parent'->>'id'
                            ),
                            'title',
                            data->'parent'->>'title'
                        )
                    )
                    WHERE data->'parent'->>'id' IS NOT NULL AND data->'parent'->>'id' != ''
                """)
                
                # Update children references
                cur.execute("""
                    UPDATE issues_simplified
                    SET data = jsonb_set(
                        data,
                        '{children}',
                        (
                            SELECT jsonb_agg(
                                jsonb_build_object(
                                    'id',
                                    COALESCE(
                                        (SELECT data->>'id'
                                        FROM issues_simplified child
                                        WHERE child.id = child_elem->>'id'
                                        LIMIT 1),
                                        child_elem->>'id'
                                    ),
                                    'title',
                                    child_elem->>'title'
                                )
                            )
                            FROM jsonb_array_elements(data->'children') child_elem
                        )
                    )
                    WHERE jsonb_array_length(data->'children') > 0
                """)
                
                conn.commit()
                logger.info("Successfully updated issue references to Linear format")
                
                # Update the same in embeddings_simplified table
                cur.execute("""
                    UPDATE embeddings_simplified e
                    SET data = i.data
                    FROM issues_simplified i
                    WHERE e.issue_id = i.id
                """)
                
                conn.commit()
                logger.info("Successfully synced updated references to embeddings table")
                
                return True
    except Exception as e:
        logger.error(f"Error updating issue references: {str(e)}")
        return False

def run_daily_update():
    """Run the daily update task - clear and reimport data."""
    start_time = datetime.now()
    logger.info(f"Starting daily update at {start_time}")
    
    # Step 1: Clear existing database tables (but preserve embeddings table)
    if not clear_database_tables():
        logger.error("Failed to clear database")
        return False
    
    # Step 2: Generate Linear data
    if not run_linear_data_generation():
        logger.error("Failed to generate Linear data")
        return False
    
    # Step 3: Import data to database
    if not import_data_to_database():
        logger.error("Failed to import data to database")
        return False
    
    # Step 4: Update full_context format
    if not update_full_context_in_database():
        logger.error("Failed to update full_context format")
        return False
    
    # Step 5: Update issue references to use Linear IDs
    if not update_issue_references():
        logger.error("Failed to update issue references")
        return False
    
    # Step 6: Update data field in embeddings_simplified
    if not update_embeddings_data():
        logger.error("Failed to update embeddings data")
        return False
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Daily update completed at {end_time}")
    logger.info(f"Duration: {duration}")
    
    # Get database statistics
    db_stats = linear_rag_db_import.check_existing_data()
    logger.info(f"Database now has {db_stats['issues_count']} issues")
    
    return True

def run_weekly_update():
    """Run the weekly update - generate new embeddings."""
    start_time = datetime.now()
    logger.info(f"Starting weekly update at {start_time}")
    
    # Step 1: Clear existing embeddings
    if not clear_embeddings_only():
        logger.error("Failed to clear embeddings")
        return False
    
    # Step 2: Generate new embeddings
    if not generate_embeddings():
        logger.error("Failed to generate embeddings")
        return False
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Weekly update completed at {end_time}")
    logger.info(f"Total duration: {duration}")
    
    # Get final database statistics
    db_stats = linear_rag_db_import.check_existing_data()
    logger.info("\nDatabase Statistics:")
    logger.info(f"- Issues:     {db_stats['issues_count']}")
    logger.info(f"- Embeddings: {db_stats['embeddings_count']}")
    
    if db_stats['issues_count'] > 0:
        coverage = round(db_stats['embeddings_count'] / db_stats['issues_count'] * 100, 2)
        logger.info(f"- Coverage:   {coverage}%")
    
    return True

def main():
    """Main entry point for the scheduled Linear database pipeline."""
    args = parse_arguments()
    setup_environment(args)
    
    logger.info("Setting up scheduled tasks...")
    
    # Schedule daily updates at 1 AM
    schedule.every().day.at("01:00").do(run_daily_update)
    
    # Schedule weekly updates on Monday at 2 AM
    schedule.every().monday.at("02:00").do(run_weekly_update)
    
    logger.info("Scheduled tasks:")
    logger.info("- Daily updates: Every day at 1:00 AM")
    logger.info("- Weekly updates: Every Monday at 2:00 AM")
    
    # Run immediately if requested
    if args.run_now:
        logger.info("Running pipeline immediately (--run-now flag set)")
        run_daily_update()
        logger.info("Daily update completed")
        
        # Run weekly update if flag is set
        if args.weekly:
            logger.info("Running weekly update (--weekly flag set)")
            run_weekly_update()
            logger.info("Weekly update completed")
    
    # Keep the script running
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}", exc_info=True)
            time.sleep(60)
    
    logger.info("Scheduler stopped")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 