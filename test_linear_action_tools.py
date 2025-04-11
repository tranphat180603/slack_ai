#!/usr/bin/env python3
"""
Test script for Linear action tools (createIssue and updateIssue).
This script demonstrates how to use the Linear API to create and update issues.
"""

import os
import sys
import json
import logging
import time
from dotenv import load_dotenv
from tools.tools_declaration import linear_tools
from ops_linear_db.linear_client import LinearError

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test parameters
TEAM_KEY = "OPS"
ASSIGNEE_NAME = "phat"
CYCLE_NUMBER = 42

def get_team_id(team_key):
    """Get the team ID for a team key"""
    logger.info(f"Looking up team ID for key: {team_key}")
    
    # This is a workaround since we don't have a direct method to look up team ID
    # We'll get all issues and extract the team ID from one of them
    issues = linear_tools.filterIssues(team_key=team_key, first=1)
    if not issues:
        raise ValueError(f"No issues found for team {team_key}. Cannot get team ID.")
    
    team_id = issues[0].get("team", {}).get("id")
    if not team_id:
        raise ValueError(f"Could not find team ID for team {team_key}")
    
    logger.info(f"Found team ID: {team_id} for team key: {team_key}")
    return team_id

def get_assignee_id(display_name):
    """Get the user ID for a display name"""
    logger.info(f"Looking up user ID for display name: {display_name}")
    
    # Get user information from all teams
    users = []
    for team in ["OPS", "ENG", "RES", "AI", "MKT", "PRO"]:
        try:
            team_users = linear_tools.getAllUsers(team)
            users.extend(team_users)
        except Exception as e:
            logger.warning(f"Error getting users for team {team}: {e}")
    
    # Find the user with the matching display name
    for user in users:
        if user.get("displayName", "").lower() == display_name.lower():
            user_id = user.get("id")
            logger.info(f"Found user ID: {user_id} for display name: {display_name}")
            return user_id
    
    raise ValueError(f"No user found with display name: {display_name}")

def get_cycle_id(team_key, cycle_number):
    """Get the cycle ID for a cycle number in a team"""
    logger.info(f"Looking up cycle ID for cycle number: {cycle_number} in team: {team_key}")
    
    cycles = linear_tools.getAllCycles(team_key)
    for cycle in cycles:
        if cycle.get("number") == cycle_number:
            cycle_id = cycle.get("id")
            logger.info(f"Found cycle ID: {cycle_id} for cycle number: {cycle_number}")
            return cycle_id
    
    raise ValueError(f"No cycle found with number: {cycle_number} in team: {team_key}")

def get_state_id(team_key, state_name="Todo"):
    """Get the state ID for a state name in a team"""
    logger.info(f"Looking up state ID for state name: {state_name} in team: {team_key}")
    
    # Get all states for the specific team
    states = linear_tools.getAllStates(team_key)
    
    # Debug log to see what states are available for this team
    logger.debug(f"Available states for team {team_key}: {[s.get('name') for s in states]}")
    
    # Find the state with the matching name
    for state in states:
        if state.get("name") == state_name and state.get("team", {}).get("key") == team_key:
            state_id = state.get("id")
            logger.info(f"Found state ID: {state_id} for state name: {state_name} in team: {team_key}")
            return state_id
    
    # If we can't find an exact match, use any available Todo state
    todo_states = [s for s in states if s.get("name") == state_name]
    if todo_states:
        state_id = todo_states[0].get("id")
        logger.info(f"Using alternative state ID: {state_id} for state name: {state_name}")
        return state_id
    
    # If still no match, use the first state available
    if states:
        state_id = states[0].get("id")
        logger.info(f"Using first available state ID: {state_id} with name: {states[0].get('name')}")
        return state_id
    
    raise ValueError(f"No state found for team: {team_key}")

def get_sample_issue_data():
    """Generate sample issue data for testing"""
    timestamp = int(time.time())
    
    return {
        "title": f"Test Issue {timestamp}",
        "description": "This is a test issue created by the automated test script.\n\n"
                      "## Features\n\n- Test feature 1\n- Test feature 2\n\n"
                      "## Requirements\n\n- Requirement 1\n- Requirement 2",
        "priority": 3.0,  # Medium priority
        "estimate": 2.0,  # 2 points
    }

def test_create_issue():
    """Test creating an issue"""
    try:
        # Get required IDs
        team_id = get_team_id(TEAM_KEY)
        
        # Prepare issue data with only required fields
        timestamp = int(time.time())
        issue_data = {
            "teamId": team_id,
            "title": f"Test Issue {timestamp}",
            "description": "This is a test issue created by the automated test script.\n\n"
                          "## Features\n\n- Test feature 1\n- Test feature 2\n\n"
                          "## Requirements\n\n- Requirement 1\n- Requirement 2",
            "priority": 3.0  # Medium priority
        }
        
        # Create the issue
        logger.info(f"Creating test issue with minimal data: {json.dumps(issue_data)}")
        result = linear_tools.createIssue(data=issue_data)
        
        if not result.get("success"):
            logger.error(f"Failed to create issue: {result}")
            return None
        
        issue = result.get("issue", {})
        issue_number = issue.get("number")
        logger.info(f"Successfully created issue #{issue_number}: {issue.get('title')}")
        logger.info(f"Issue details: {json.dumps(issue, indent=2)}")
        
        return issue_number
    
    except Exception as e:
        logger.error(f"Error creating issue: {e}")
        return None

def test_update_issue(issue_number):
    """Test updating an issue"""
    if not issue_number:
        logger.error("No issue number provided for update test")
        return
    
    try:
        # Prepare update data
        update_data = {
            "title": f"Updated Test Issue {int(time.time())}",
            "description": "This issue has been updated by the automated test script.\n\n"
                          "## Updated Features\n\n- Feature A\n- Feature B\n\n"
                          "## Updated Requirements\n\n- Requirement X\n- Requirement Y",
            "priority": 2.0  # High priority
        }
        
        # Update the issue
        logger.info(f"Updating issue #{issue_number} with data: {json.dumps(update_data)}")
        result = linear_tools.updateIssue(issueNumber=issue_number, data=update_data)
        
        if not result.get("success"):
            logger.error(f"Failed to update issue: {result}")
            return
        
        updated_issue = result.get("issue", {})
        logger.info(f"Successfully updated issue #{issue_number}: {updated_issue.get('title')}")
        logger.info(f"Updated issue details: {json.dumps(updated_issue, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error updating issue: {e}")

def run_tests():
    """Run all tests"""
    logger.info("Starting Linear action tools tests")
    
    # Test creating an issue
    issue_number = test_create_issue()
    
    if issue_number:
        # Test updating the created issue
        test_update_issue(issue_number)
    
    logger.info("Linear action tools tests completed")

if __name__ == "__main__":
    run_tests() 