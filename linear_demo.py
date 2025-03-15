import os
import json
from linear_client import LinearClient
import logging
from datetime import datetime
from dotenv import load_dotenv
import pathlib

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

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
    data_dir = pathlib.Path("1_month_data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize data structure
    all_data = {
        "teams": [],
        "cycles": [],
        "issues": []
    }
    
    print("\nFetching current cycles for all teams...")
    successful_teams = 0
    
    for team in teams:
        print(f"Checking team: {team.name} ({team.key})")
        
        # Get the current active cycle for this team
        team_cycle = client.get_cycle_data(team_key=team.key, current=True)
        
        if "error" not in team_cycle and "cycle" in team_cycle:
            current_cycle_number = team_cycle['cycle']['number']
            current_cycle_name = team_cycle['cycle']['name'] or f"Cycle {current_cycle_number}"
            print(f"  ✓ Active cycle found: {current_cycle_name}")
            
            cycle_id = team_cycle['cycle']['id']
            print(f"  ↳ Getting detailed data for {current_cycle_name}")
            comprehensive_data = client.get_comprehensive_cycle_data(cycle_id)
            
            if "error" not in comprehensive_data:
                # Format the data for this cycle
                formatted_data = format_data_for_rag_import(comprehensive_data, team.key)
                
                # Add to our consolidated data structure
                all_data["teams"].extend(formatted_data["teams"])
                all_data["cycles"].extend(formatted_data["cycles"])
                all_data["issues"].extend(formatted_data["issues"])
                
                cycle_issues_count = len(formatted_data['issues'])
                print(f"    ✓ Added data for {current_cycle_name} - {cycle_issues_count} issues")
                successful_teams += 1
            else:
                print(f"    ✗ Error getting detailed data for {current_cycle_name}: {comprehensive_data['error']}")
        else:
            error_msg = team_cycle.get("error", "No active cycle")
            print(f"  ✗ No active cycle for {team.key}: {error_msg}")
    
    # Remove duplicate teams by ID
    unique_teams = {}
    for team in all_data["teams"]:
        if team["id"] not in unique_teams:
            unique_teams[team["id"]] = team
    all_data["teams"] = list(unique_teams.values())
    
    # Remove duplicate cycles by ID
    unique_cycles = {}
    for cycle in all_data["cycles"]:
        if cycle["id"] not in unique_cycles:
            unique_cycles[cycle["id"]] = cycle
    all_data["cycles"] = list(unique_cycles.values())
    
    # Remove duplicate issues by ID
    unique_issues = {}
    for issue in all_data["issues"]:
        if issue["id"] not in unique_issues:
            unique_issues[issue["id"]] = issue
    all_data["issues"] = list(unique_issues.values())
    
    # Extract and deduplicate employees from issues
    all_data["employees"] = []
    unique_employees = {}
    for issue in all_data["issues"]:
        assignee = issue.get("assignee", {})
        if assignee and assignee.get("id"):
            employee_id = assignee.get("id")
            if employee_id and employee_id not in unique_employees:
                unique_employees[employee_id] = {
                    "id": employee_id,
                    "name": assignee.get("name", "")
                }
    
    # Add deduplicated employees to the output
    all_data["employees"] = list(unique_employees.values())
    print(f"Added {len(all_data['employees'])} unique employees to the data")
    
    # Save all data to a single file
    data_file = data_dir / "current_cycles_data.json"
    with open(data_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"\nSaved data with {len(all_data['issues'])} issues and {len(all_data['cycles'])} cycles to {data_file}")
    
    # Print some details about the cycles saved
    print("\nCycles included in the data:")
    cycle_info = []
    for cycle in all_data["cycles"]:
        team_key = cycle.get("team_key", "Unknown")
        cycle_name = cycle.get("name", "Unknown") or f"Cycle {cycle.get('number')}"
        cycle_info.append(f"{team_key}: {cycle_name}")
    
    if cycle_info:
        # Sort cycle info to make it more readable
        cycle_info.sort()
        for info in cycle_info:
            print(f"  - {info}")
    
    print(f"\nSuccessfully retrieved detailed data for {successful_teams} teams")
    print(f"Data has been saved to: {data_file}")


def format_data_for_rag_import(data, team_key=None):
    """
    Format the cycle data to match the expected structure in linear_rag_import.py.
    
    Args:
        data: The cycle data to format
        team_key: The key of the team if known
        
    Returns:
        A dictionary with the properly formatted data
    """
    formatted_data = {
        "teams": [],
        "cycles": [],
        "issues": []
    }
    
    # Format teams
    for team in data.get('teams', []):
        formatted_team = {
            "id": team.get('id', ''),
            "name": team.get('name', ''),
            "key": team.get('key', ''),
            "description": team.get('description', '')
        }
        formatted_data["teams"].append(formatted_team)
    
    # First pass: collect cycles with non-null names along with their date ranges
    named_cycles_by_date = {}
    for cycle in data.get('cycles', []):
        cycle_name = cycle.get('name')
        if cycle_name:  # Only store cycles with actual names
            starts_at = cycle.get('starts_at', '')
            ends_at = cycle.get('ends_at', '')
            date_key = f"{starts_at}_{ends_at}"
            named_cycles_by_date[date_key] = cycle_name
    
    # Format cycles
    for cycle in data.get('cycles', []):
        cycle_number = cycle.get('number', 0)
        cycle_name = cycle.get('name')
        
        # If cycle name is null, try to find a match from a cycle with the same dates
        if not cycle_name:
            starts_at = cycle.get('starts_at', '')
            ends_at = cycle.get('ends_at', '')
            date_key = f"{starts_at}_{ends_at}"
            if date_key in named_cycles_by_date:
                cycle_name = named_cycles_by_date[date_key]
                print(f"      Fixed null cycle name for team {cycle.get('team_key', team_key)} using name: {cycle_name}")
        
        # If still no name, use default naming
        if not cycle_name:
            cycle_name = f"Cycle {cycle_number}"
        
        formatted_cycle = {
            "id": cycle.get('id', ''),
            "name": cycle_name,
            "number": cycle_number,
            "team_key": cycle.get('team_key', team_key),
            "starts_at": cycle.get('starts_at', ''),
            "ends_at": cycle.get('ends_at', ''),
            "progress": cycle.get('progress', 0)
        }
        formatted_data["cycles"].append(formatted_cycle)
    
    # If the data contains a single cycle (not in cycles array), add it too
    cycle_id = None
    if "cycle" in data and data["cycle"]:
        cycle = data["cycle"]
        cycle_id = cycle.get('id', '')  # Store the cycle ID to link with issues
        cycle_number = cycle.get("number", 0)
        cycle_name = cycle.get("name")
        
        # If cycle name is null, try to find a match based on dates
        if not cycle_name:
            starts_at = cycle.get('starts_at', '')
            ends_at = cycle.get('ends_at', '')
            date_key = f"{starts_at}_{ends_at}"
            if date_key in named_cycles_by_date:
                cycle_name = named_cycles_by_date[date_key]
                print(f"      Fixed null cycle name for current cycle using name: {cycle_name}")
            else:
                # If we don't have exact date matches but we already processed cycles,
                # try to find a cycle with the same number
                for processed_cycle in formatted_data["cycles"]:
                    if processed_cycle.get("number") == cycle_number and processed_cycle.get("name"):
                        cycle_name = processed_cycle.get("name")
                        print(f"      Fixed null cycle name using cycle with same number: {cycle_name}")
                        break
                
        # If still no name, use default naming
        if not cycle_name:
            cycle_name = f"Cycle {cycle_number}"
        
        formatted_cycle = {
            "id": cycle_id,
            "name": cycle_name,
            "number": cycle_number,
            "team_key": team_key,
            "starts_at": cycle.get('starts_at', ''),
            "ends_at": cycle.get('ends_at', ''),
            "progress": cycle.get('progress', 0)
        }
        
        # Check if this cycle already exists
        existing_cycle = next((c for c in formatted_data["cycles"] if c["id"] == formatted_cycle["id"]), None)
        if not existing_cycle:
            formatted_data["cycles"].append(formatted_cycle)
            print(f"      Added {cycle_name} to formatted data")
    
    # Format issues
    for issue in data.get('issues', []):
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
            "key": issue.get('team', {}).get('key', ''),
            "name": issue.get('team', {}).get('name', '')
        }
        
        # Format parent
        parent = {
            "id": issue.get('parent', {}).get('id', '')
        }
        
        # Format cycle - use the cycle_id from the parent data structure
        # since issues from get_comprehensive_cycle_data belong to that cycle
        cycle = None
        if issue.get('cycle'):
            cycle = {
                "id": issue.get('cycle', {}).get('id', '')
            }
        elif cycle_id:  # If no cycle in issue but we have a cycle ID from the parent data
            cycle = {
                "id": cycle_id
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
        
        formatted_issue = {
            "id": issue.get('id', ''),
            "title": issue.get('title', ''),
            "description": issue.get('description', ''),
            "state": state,
            "state_type": state_type,
            "assignee": assignee,
            "team": team,
            "parent": parent,
            "cycle": cycle,  # This will now contain the cycle ID
            "priority": issue.get('priority', ''),
            "estimate": issue.get('estimate', None),
            "created_at": issue.get('created_at', ''),
            "updated_at": issue.get('updated_at', ''),
            "completed_at": issue.get('completed_at', ''),
            "comments": comments,
            "children": issue.get('children', [])
        }
        
        formatted_data["issues"].append(formatted_issue)
    
    return formatted_data


if __name__ == "__main__":
    main() 