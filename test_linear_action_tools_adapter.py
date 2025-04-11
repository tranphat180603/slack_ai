#!/usr/bin/env python3
"""
Test script for Linear action tools (createIssue and updateIssue) using parameter adapters.
This script demonstrates how to use the Linear API with simpler parameter formats.
"""

import os
import json
import logging
import time
from dotenv import load_dotenv
from tools.tools_declaration import linear_tools, LinearParameterAdapter
from ops_linear_db.linear_client import LinearError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test parameters
TEAM_KEY = "OPS"
ASSIGNEE_NAME = "phat"
CYCLE_NUMBER = 42

def test_create_issue_with_adapter():
    """Test creating an issue using adapter-friendly parameters"""
    try:
        # First, find the state name (e.g., "Todo")
        states = linear_tools.getAllStates(TEAM_KEY)
        todo_state = next((state for state in states if state.get("name") == "Todo"), None)
        if not todo_state:
            logger.warning("Todo state not found, using first available state")
            todo_state = states[0] if states else {"name": "Todo"}
        
        state_name = todo_state.get("name")
        
        # Get cycle name
        cycles = linear_tools.getAllCycles(TEAM_KEY)
        target_cycle = next((cycle for cycle in cycles if cycle.get("number") == CYCLE_NUMBER), None)
        if not target_cycle:
            logger.warning(f"Cycle {CYCLE_NUMBER} not found, test will fail")
            cycle_name = f"Cycle {CYCLE_NUMBER}"
        else:
            cycle_name = target_cycle.get("name")
        
        # Create issue using direct parameters (no need to look up IDs)
        timestamp = int(time.time())
        
        # Prepare parameters
        params = {
            "teamKey": TEAM_KEY,  # Note: using teamKey not teamId
            "title": f"Adapter Test Issue {timestamp}",
            "description": "This is a test issue created with the adapter.\n\n"
                          "## Features\n\n- Test feature 1\n- Test feature 2\n\n"
                          "## Requirements\n\n- Requirement 1\n- Requirement 2",
            "priority": 3.0,  # Medium priority
            "estimate": 2.0,  # 2 points
            "assignee_name": ASSIGNEE_NAME,  # Note: using display name not ID
            "state_name": state_name,
            "cycle_name": cycle_name
        }
        
        # Log the raw parameters
        logger.info(f"Creating issue with raw parameters: {json.dumps(params)}")
        
        # Show how parameters would be adapted
        adapted = LinearParameterAdapter.adapt_create_issue(params)
        logger.info(f"Parameters would be adapted to: {json.dumps(adapted)}")
        
        # Create the issue
        # Note: In a real implementation, the adapter would handle ID lookups
        # For this test, we'll manually create the proper data structure
        
        # Manually adapt parameters (in a real implementation, this would be handled by the adapter)
        from ops_linear_db.linear_client import LinearClient
        linear_client = LinearClient(os.getenv("LINEAR_API_KEY"))
        
        # Look up team ID
        issues = linear_tools.filterIssues(team_key=TEAM_KEY, first=1)
        if not issues:
            raise ValueError(f"No issues found for team {TEAM_KEY}")
        team_id = issues[0].get("team", {}).get("id")
        
        # Look up user ID
        users = []
        for team in ["OPS", "ENG", "RES", "AI", "MKT", "PRO"]:
            try:
                team_users = linear_tools.getAllUsers(team)
                users.extend(team_users)
            except Exception as e:
                pass
        
        user = next((u for u in users if u.get("displayName", "").lower() == ASSIGNEE_NAME.lower()), None)
        if not user:
            raise ValueError(f"No user found with display name: {ASSIGNEE_NAME}")
        assignee_id = user.get("id")
        
        # Create data object
        data = {
            "teamId": team_id,
            "title": params["title"],
            "description": params["description"],
            "priority": params["priority"],
            "estimate": params["estimate"],
            "assigneeId": assignee_id
        }
        
        # Look up state ID
        state = next((s for s in states if s.get("name") == state_name), None)
        if state:
            data["stateId"] = state.get("id")
        
        # Look up cycle ID
        cycle = next((c for c in cycles if c.get("number") == CYCLE_NUMBER), None)
        if cycle:
            data["cycleId"] = cycle.get("id")
        
        # Create issue
        result = linear_client.createIssue(data)
        
        if not result or not result.get("issue"):
            logger.error(f"Failed to create issue: {result}")
            return None
        
        issue = result.get("issue", {})
        issue_number = issue.get("number")
        logger.info(f"Successfully created issue #{issue_number}: {issue.get('title')}")
        logger.info(f"Issue details: {json.dumps(issue, indent=2) if issue else 'None'}")
        
        return issue_number
    
    except Exception as e:
        logger.error(f"Error creating issue: {e}")
        return None

def test_update_issue_with_adapter(issue_number):
    """Test updating an issue using adapter-friendly parameters"""
    if not issue_number:
        logger.error("No issue number provided for update test")
        return
    
    try:
        # Prepare parameters
        timestamp = int(time.time())
        params = {
            "issue_number": issue_number,  # Note: using issue_number not issueNumber
            "title": f"Updated Adapter Test Issue {timestamp}",
            "description": "This issue has been updated with the adapter.\n\n"
                          "## Updated Features\n\n- Feature A\n- Feature B\n\n"
                          "## Updated Requirements\n\n- Requirement X\n- Requirement Y",
            "priority": 2.0  # High priority
        }
        
        # Log the raw parameters
        logger.info(f"Updating issue with raw parameters: {json.dumps(params)}")
        
        # Show how parameters would be adapted
        adapted = LinearParameterAdapter.adapt_update_issue(params)
        logger.info(f"Parameters would be adapted to: {json.dumps(adapted)}")
        
        # In a real implementation with a complete adapter, we would use:
        # result = linear_tools.updateIssue(**params)
        
        # For this test, we'll use the adapted parameters directly
        from ops_linear_db.linear_client import LinearClient
        linear_client = LinearClient(os.getenv("LINEAR_API_KEY"))
        
        result = linear_client.updateIssue(
            issueNumber=issue_number,
            data={
                "title": params["title"],
                "description": params["description"],
                "priority": params["priority"]
            }
        )
        
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
    logger.info("Starting Linear action tools adapter tests")
    
    # Test creating an issue with adapter
    issue_number = test_create_issue_with_adapter()
    
    if issue_number:
        # Test updating the created issue with adapter
        test_update_issue_with_adapter(issue_number)
    
    logger.info("Linear action tools adapter tests completed")

if __name__ == "__main__":
    run_tests() 