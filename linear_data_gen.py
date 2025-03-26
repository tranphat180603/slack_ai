import os
import json
from linear_client import LinearClient
import logging
from datetime import datetime
from dotenv import load_dotenv
import pathlib
import re
from typing import List, Dict, Any

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("linear_data_gen")

def main():
    # Get API key from environment variable
    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key:
        print("ERROR: Please set the LINEAR_API_KEY environment variable")
        return
    
    # Initialize the Linear client
    client = LinearClient(api_key)
    print("Linear client initialized")
    
    # Step 1: Get all teams
    teams = client.get_all_teams()
    if not teams:
        print("No teams found in your Linear workspace")
        return
    
    # Filter out teams we don't want to include
    teams_to_remove = ["MGT", "DAO2"]
    teams = [team for team in teams if team.key not in teams_to_remove]
    
    print(f"Found {len(teams)} teams in your workspace")
    
    # Create a directory for storing data
    data_dir = pathlib.Path("cycles_data")
    data_dir.mkdir(exist_ok=True)
    
    # First, get reference cycle numbers from Research team
    research_cycles = get_research_team_cycles(client)
    if not research_cycles:
        print("Could not get cycle names from Research team. Using default naming.")
    
    print(f"Using reference cycle names: Active={research_cycles['active']}, Recent={research_cycles['recent']}")
    
    # Step 2: Collect all cycles from all teams
    print("\nFetching last 4 cycles for all teams...")
    all_cycles = []
    
    for team in teams:
        print(f"Checking team: {team.name} ({team.key})")
        
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
                cycle_number = extract_cycle_number(research_cycles['active'])
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
                print(f"  ✓ Active cycle found: {original_name} → {research_cycles['active']}")
            
            # Add recent cycles
            recent_cycles = team_cycles_data.get('recent_cycles', [])
            for i, cycle in enumerate(recent_cycles):
                if i < len(research_cycles['recent']):
                    # Use the Research team's cycle name for this position
                    original_name = cycle.get('name', '')
                    cycle['name'] = research_cycles['recent'][i]
                    
                    # Extract cycle number from name
                    cycle_number = extract_cycle_number(research_cycles['recent'][i])
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
                    print(f"  ✓ Recent cycle {i+1}: {original_name} → {research_cycles['recent'][i]}")
            
            print(f"  ✓ Retrieved {len(recent_cycles)} historical cycles")
        else:
            error_msg = team_cycles_data.get("error", "Unknown error")
            print(f"  ✗ Error getting cycles for {team.key}: {error_msg}")
    
    print(f"\nCollected {len(all_cycles)} cycles from all teams")
    
    # Step 3: Group cycles by standardized names (not by date)
    cycle_groups = group_cycles_by_name(all_cycles, research_cycles)
    print(f"Grouped into {len(cycle_groups)} distinct cycle periods")
    
    # Step 4: Ensure we have correct week labeling (active cycle is week_4)
    # Sort groups by cycle number, highest first
    cycle_groups.sort(key=lambda g: g['cycle_number'], reverse=True)
    
    # Assign week numbers, ensuring active cycle is week_4
    for i, group in enumerate(cycle_groups[:4]):
        week_number = 4 - i  # Active cycle (highest number) is week_4, count down from there
        group['week_label'] = f"week_{week_number}"
        print(f"Week {week_number}: {group['standardized_name']} ({group['starts_at']} to {group['ends_at']})")
    
    # Step 5: Process only the 4 most recent cycle groups
    cycles_to_process = cycle_groups[:4]
    for week_group in cycles_to_process:
        process_cycle_group(client, week_group, data_dir)
    
    print("\nSummary of data collection:")
    for group in cycles_to_process:
        print(f"  - {group['week_label']}: {group['standardized_name']} - {len(group.get('processed_issues', []))} issues")
    
    print(f"\nData has been saved to: {data_dir}")


def extract_cycle_number(cycle_name):
    """Extract the numeric part from a cycle name (e.g., 'Cycle 42' -> 42)."""
    if not cycle_name:
        return 0
    
    match = re.search(r'Cycle\s+(\d+)', cycle_name)
    if match:
        return int(match.group(1))
    return 0


def get_research_team_cycles(client):
    """
    Get cycle names from Research team to use as a reference.
    If Research team data is invalid or inconsistent, fall back to other teams.
    """
    # Path to store the last known cycle number
    cycle_history_file = pathlib.Path("cycles_data/last_cycle.json")
    last_cycle_number = 0
    
    # Try to load the previous week's cycle number
    if cycle_history_file.exists():
        try:
            with open(cycle_history_file, 'r') as f:
                history_data = json.load(f)
                last_cycle_number = history_data.get('last_cycle_number', 0)
                print(f"Found last cycle number: {last_cycle_number}")
        except Exception as e:
            print(f"Error loading last cycle number: {str(e)}")
    
    # First try RES team (primary source)
    try:
        research_cycles_data = client.get_team_recent_cycles(team_key="RES", limit=4)
        
        if "error" not in research_cycles_data:
            active_cycle = research_cycles_data.get('active_cycle', {})
            active_name = active_cycle.get('name', '') if active_cycle else ""
            active_cycle_number = extract_cycle_number(active_name)
            
            # Validate the active cycle number against last known cycle
            is_valid = True
            if last_cycle_number > 0:
                # Check if the cycle number has incremented by exactly 1
                if active_cycle_number != last_cycle_number + 1:
                    print(f"Warning: Research team cycle number ({active_cycle_number}) doesn't follow expected sequence (expected {last_cycle_number + 1})")
                    is_valid = False
            
            if is_valid:
                recent_cycles = research_cycles_data.get('recent_cycles', [])
                recent_names = [cycle.get('name', '') for cycle in recent_cycles]
                
                # Save current cycle number for next time
                try:
                    with open(cycle_history_file, 'w') as f:
                        json.dump({'last_cycle_number': active_cycle_number}, f)
                    print(f"Saved current cycle number ({active_cycle_number}) for next run")
                except Exception as e:
                    print(f"Error saving cycle number: {str(e)}")
                
                # Ensure we have at least 3 names for the recent cycles
                while len(recent_names) < 3:
                    cycle_num = extract_cycle_number(recent_names[-1] if recent_names else active_name) - 1
                    recent_names.append(f"Cycle {cycle_num}")
                
                print(f"Using Research team cycles: Active={active_name}, Recent={recent_names[:3]}")
                return {
                    "active": active_name,
                    "recent": recent_names[:3]  # Only use the first 3 recent cycles
                }
            
            # If we get here, the Research team data is invalid, so we'll try other teams
            print("Research team data appears invalid, trying alternative teams...")
    except Exception as e:
        print(f"Error getting Research team cycles: {str(e)}")
    
    # Fallback teams to try if Research team data is invalid
    fallback_teams = ["MKT", "PRO", "OPS"]
    
    for team_key in fallback_teams:
        try:
            print(f"Trying to get cycle data from {team_key} team...")
            team_cycles_data = client.get_team_recent_cycles(team_key=team_key, limit=4)
            
            if "error" not in team_cycles_data:
                active_cycle = team_cycles_data.get('active_cycle', {})
                active_name = active_cycle.get('name', '') if active_cycle else ""
                active_cycle_number = extract_cycle_number(active_name)
                
                # Validate the active cycle number against last known cycle
                is_valid = True
                if last_cycle_number > 0:
                    # The cycle should be either the same or incremented by 1
                    if active_cycle_number != last_cycle_number and active_cycle_number != last_cycle_number + 1:
                        print(f"Warning: {team_key} team cycle number ({active_cycle_number}) doesn't follow expected sequence")
                        is_valid = False
                
                if is_valid and active_name:
                    recent_cycles = team_cycles_data.get('recent_cycles', [])
                    recent_names = [cycle.get('name', '') for cycle in recent_cycles]
                    
                    # Save current cycle number for next time if it's newer
                    if active_cycle_number > last_cycle_number:
                        try:
                            with open(cycle_history_file, 'w') as f:
                                json.dump({'last_cycle_number': active_cycle_number}, f)
                            print(f"Saved current cycle number ({active_cycle_number}) from {team_key} team for next run")
                        except Exception as e:
                            print(f"Error saving cycle number: {str(e)}")
                    
                    # Ensure we have at least 3 names for the recent cycles
                    while len(recent_names) < 3:
                        cycle_num = extract_cycle_number(recent_names[-1] if recent_names else active_name) - 1
                        recent_names.append(f"Cycle {cycle_num}")
                    
                    print(f"Using {team_key} team cycles: Active={active_name}, Recent={recent_names[:3]}")
                    return {
                        "active": active_name,
                        "recent": recent_names[:3]  # Only use the first 3 recent cycles
                    }
        except Exception as e:
            print(f"Error getting {team_key} team cycles: {str(e)}")
    
    # If we get here, all attempts failed - create reasonable defaults based on last known cycle
    if last_cycle_number > 0:
        print(f"All team lookups failed, using last known cycle number ({last_cycle_number}) to generate defaults")
        # Assume we're now in the next cycle
        current_cycle = last_cycle_number + 1
        active_name = f"Cycle {current_cycle}"
        recent_names = [f"Cycle {current_cycle-i}" for i in range(1, 4)]
        
        return {
            "active": active_name,
            "recent": recent_names
        }
    
    # Ultimate fallback if we have no history and all team lookups failed
    print("WARNING: Could not determine current cycle from any team or history. Using emergency defaults.")
    return {
        "active": "Cycle 1",
        "recent": ["Cycle 0", "Cycle -1", "Cycle -2"]
    }


def group_cycles_by_name(cycles, reference_cycles):
    """
    Group cycles by standardized names from Research team.
    Returns a list of cycle groups.
    """
    # Create mapping for standardized names
    cycle_name_groups = {}
    
    # First, create entries for each reference cycle
    all_cycle_names = [reference_cycles['active']] + reference_cycles['recent']
    for cycle_name in all_cycle_names:
        cycle_name_groups[cycle_name] = {
            'cycles': [],
            'standardized_name': cycle_name,
            'cycle_number': extract_cycle_number(cycle_name),
            'starts_at': '',
            'ends_at': '',
            'teams': []
        }
    
    # Add cycles to the appropriate groups
    for cycle in cycles:
        cycle_name = cycle.get('name', '')
        if cycle_name in cycle_name_groups:
            # Add to existing group
            group = cycle_name_groups[cycle_name]
            group['cycles'].append(cycle)
            
            # Update start and end dates
            if cycle.get('starts_at') and (not group['starts_at'] or cycle['starts_at'] < group['starts_at']):
                group['starts_at'] = cycle['starts_at']
            if cycle.get('ends_at') and (not group['ends_at'] or cycle['ends_at'] > group['ends_at']):
                group['ends_at'] = cycle['ends_at']
            
            # Add team if not already in list
            if cycle.get('team_key') and cycle['team_key'] not in group['teams']:
                group['teams'].append(cycle['team_key'])
        
        # No need for else - we use only the predefined cycle names
    
    # Convert to list and return
    result = list(cycle_name_groups.values())
    
    # Remove any empty groups (no cycles)
    result = [group for group in result if group['cycles']]
    
    return result


def process_cycle_group(client, cycle_group, data_dir):
    """Process a group of cycles and save as a JSON file."""
    week_label = cycle_group.get('week_label', 'unknown_week')
    standardized_name = cycle_group.get('standardized_name', 'Unknown Cycle')
    
    print(f"\nProcessing {week_label} ({standardized_name})...")
    
    # Initialize data structure for the week
    week_data = {
        "issues": []
    }
    
    # Dictionary to track named cycles by date ranges for the formatter
    named_cycles_by_date = {}
    for cycle in cycle_group.get('cycles', []):
        date_key = f"{cycle.get('starts_at', '')}_{cycle.get('ends_at', '')}"
        named_cycles_by_date[date_key] = standardized_name
    
    # Process each cycle in the group
    all_issues = []
    processed_cycle_count = 0
    
    for cycle in cycle_group.get('cycles', []):
        cycle_id = cycle.get('id')
        if not cycle_id:
            continue
        
        print(f"  ↳ Getting detailed data for {cycle.get('name', 'Unnamed cycle')} (Team: {cycle.get('team_key', 'Unknown')})")
        comprehensive_data = client.get_comprehensive_cycle_data(cycle_id)
        
        if "error" not in comprehensive_data:
            # Add the standardized cycle name to the comprehensive data
            # This ensures all issues in this week use the same standardized cycle name
            if "cycle" not in comprehensive_data:
                comprehensive_data["cycle"] = {}
            comprehensive_data["cycle"]["standardized_name"] = standardized_name
            
            # Format the data for the simplified schema
            issues = format_issues_for_simplified_schema(comprehensive_data, cycle.get('team_key'), named_cycles_by_date)
            
            # Add to our collection
            all_issues.extend(issues)
            processed_cycle_count += 1
            print(f"    ✓ Added {len(issues)} issues from {cycle.get('team_key', 'Unknown')}")
        else:
            print(f"    ✗ Error getting detailed data: {comprehensive_data['error']}")
    
    # Remove duplicate issues by ID
    unique_issues = {}
    for issue in all_issues:
        if issue["id"] not in unique_issues:
            unique_issues[issue["id"]] = issue
    
    week_data["issues"] = list(unique_issues.values())
    cycle_group['processed_issues'] = week_data["issues"]  # Store for summary reporting
    
    # Save week data to JSON file
    data_file = data_dir / f"{week_label}.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(week_data, f, indent=2)
    
    print(f"✓ Saved {len(week_data['issues'])} issues for {week_label} to {data_file}")
    print(f"  Processed {processed_cycle_count}/{len(cycle_group.get('cycles', []))} cycles")
    
    # Save a summary of teams and cycles in this week
    summary_data = {
        "week": week_label,
        "name": standardized_name,
        "starts_at": cycle_group.get('starts_at', ''),
        "ends_at": cycle_group.get('ends_at', ''),
        "teams": cycle_group.get('teams', []),
        "issue_count": len(week_data["issues"]),
        "teams_detail": [
            {
                "team": cycle.get('team_key', ''),
                "cycle_name": cycle.get('name', ''),
                "is_active": cycle.get('is_active', False)
            }
            for cycle in cycle_group.get('cycles', [])
        ]
    }
    
    summary_file = data_dir / f"{week_label}_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2)


def format_issues_for_simplified_schema(data: Dict[str, Any], team_key: str, named_cycles_by_date: Dict[str, str]) -> List[Dict[str, Any]]:
    """Format issues from comprehensive cycle data for simplified schema."""
    formatted_issues = []
    
    # Get the standardized cycle name from the parent data
    standardized_cycle_name = data.get("cycle", {}).get("standardized_name", "")
    
    # Create a mapping of UUID to Linear ID for all issues in this batch
    id_mapping = {}
    for issue in data.get('issues', []):
        if not issue:
            continue
            
        # Get team key from issue or use passed team_key
        issue_team_key = issue.get('team', {}).get('key', team_key or '')
        
        # Get issue number directly from the issue data
        issue_number = issue.get('number')
        
        # If number not available, try to extract from description or title
        if not issue_number:
            description = issue.get('description', '')
            if description:
                # Look for Linear issue URL pattern
                match = re.search(r'linear\.app/[^/]+/issue/([^/]+)/', description)
                if match:
                    issue_number = match.group(1)
            
            # If not found in description, try to extract from title
            if not issue_number:
                title = issue.get('title', '')
                match = re.search(r'\[([A-Z]+-\d+)\]', title)
                if match:
                    issue_number = match.group(1)
        
        # Create Linear ID
        uuid = issue.get('id', '')
        linear_id = f"{issue_team_key}-{issue_number}" if issue_team_key and issue_number else uuid
        id_mapping[uuid] = linear_id
    
    # Process all issues
    for issue in data.get('issues', []):
        if not issue:
            continue
            
        # Format state
        state = issue.get('state', '')
        state_type = ''
        if isinstance(state, dict):
            state_type = state.get('type', '')
            state = state.get('name', '')
        
        # Format assignee
        assignee = {
            "id": issue.get('assignee', {}).get('id', ''),
            "name": issue.get('assignee', {}).get('name', '')
        }
        
        # Format team
        team = {
            "id": issue.get('team', {}).get('id', ''),
            "key": issue.get('team', {}).get('key', team_key or ''),
            "name": issue.get('team', {}).get('name', '')
        }
        
        # Format parent using Linear ID if available
        parent_uuid = issue.get('parent', {}).get('id', '')
        parent = {
            "id": id_mapping.get(parent_uuid, parent_uuid),  # Use mapped Linear ID or fallback to UUID
            "title": issue.get('parent', {}).get('title', '')
        }
        
        # Format cycle using the standardized name
        cycle_info = {
            "id": "",
            "name": standardized_cycle_name if standardized_cycle_name else cycle.get("name", "Unnamed Cycle")
        }
        
        # Format comments
        comments = []
        for comment in issue.get('comments', []):
            formatted_comment = {
                "id": comment.get('id', ''),
                "body": comment.get('body', ''),
                "created_at": comment.get('created_at', ''),
                "user": comment.get('user', '')
            }
            comments.append(formatted_comment)
        
        # Format children using Linear IDs
        children = []
        for child in issue.get('children', []):
            if isinstance(child, dict):
                child_uuid = child.get('id', '')
                formatted_child = {
                    "id": id_mapping.get(child_uuid, child_uuid),  # Use mapped Linear ID or fallback to UUID
                    "title": child.get('title', '')
                }
                children.append(formatted_child)
        
        # Get issue number directly or extract from description/title
        issue_number = issue.get('number')
        if not issue_number:
            description = issue.get('description', '')
            if description:
                match = re.search(r'linear\.app/[^/]+/issue/([^/]+)/', description)
                if match:
                    issue_number = match.group(1)
            
            if not issue_number:
                title = issue.get('title', '')
                match = re.search(r'\[([A-Z]+-\d+)\]', title)
                if match:
                    issue_number = match.group(1)
        
        # Create issue data
        issue_data = {
            "id": f"{team['key']}-{issue_number}" if team['key'] and issue_number else issue.get('id', ''),
            "title": issue.get('title', ''),
            "description": issue.get('description', ''),
            "state": state,
            "state_type": state_type,
            "assignee": assignee,
            "team": team,
            "parent": parent,
            "cycle": cycle_info,
            "priority": issue.get('priority', ''),
            "estimate": issue.get('estimate', None),
            "created_at": issue.get('created_at', ''),
            "updated_at": issue.get('updated_at', ''),
            "completed_at": issue.get('completed_at', ''),
            "comments": comments,
            "children": children
        }
        
        # Create full context with only title and description
        full_context = f"Title: {issue.get('title', '')}\n"
        if issue.get('description'):
            full_context += f"\nDescription:\n{issue.get('description')}"
        
        formatted_issue = {
            "id": issue_data["id"],
            "data": issue_data,
            "full_context": full_context
        }
        formatted_issues.append(formatted_issue)
    
    return formatted_issues


def create_full_context(issue_data):
    """Create a human-readable full context string from issue data."""
    
    # Basic issue information
    context = f"Issue: {issue_data['title']}\n"
    context += f"State: {issue_data['state']}\n"
    
    # Team and assignee
    team_info = f"{issue_data['team']['name']} ({issue_data['team']['key']})" if issue_data['team']['name'] else issue_data['team']['key']
    context += f"Team: {team_info}\n"
    
    if issue_data['assignee']['name']:
        context += f"Assignee: {issue_data['assignee']['name']}\n"
    
    # Cycle
    if issue_data['cycle'] and issue_data['cycle']['name']:
        context += f"Cycle: {issue_data['cycle']['name']}\n"
    
    # Priority
    if issue_data['priority']:
        context += f"Priority: {issue_data['priority']}\n"
    
    # Parent issue
    if issue_data['parent']['id']:
        context += f"Parent: {issue_data['parent']['title']} (ID: {issue_data['parent']['id']})\n"
    
    # Description
    context += f"\nDescription:\n{issue_data['description'] or 'No description provided.'}\n"
    
    # Comments
    if issue_data['comments']:
        context += f"\nComments ({len(issue_data['comments'])}):\n"
        for comment in issue_data['comments']:
            user = comment['user'] if isinstance(comment['user'], str) else comment['user'].get('name', 'Unknown')
            created_at = comment['created_at'] if comment['created_at'] else 'Unknown date'
            context += f"- {user} ({created_at}):\n  {comment['body']}\n\n"
    
    # Children
    if issue_data['children']:
        context += f"\nSub-issues ({len(issue_data['children'])}):\n"
        for child in issue_data['children']:
            context += f"- {child['title']} (ID: {child['id']})\n"
    
    # Dates
    context += f"\nCreated: {issue_data['created_at']}\n"
    if issue_data['updated_at']:
        context += f"Updated: {issue_data['updated_at']}\n"
    if issue_data['completed_at']:
        context += f"Completed: {issue_data['completed_at']}\n"
    
    return context


if __name__ == "__main__":
    main() 