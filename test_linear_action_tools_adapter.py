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
        
        # Test the getUserByName function directly to debug
        logger.info(f"Testing getUserByName function for assignee '{ASSIGNEE_NAME}'")
        from ops_linear_db.linear_client import LinearClient
        linear_client = LinearClient(os.getenv("LINEAR_API_KEY"))
        user = linear_client.getUserByName(ASSIGNEE_NAME)
        if user:
            logger.info(f"Found user directly: {user.get('displayName')} with ID: {user.get('id')}")
        else:
            logger.warning(f"User '{ASSIGNEE_NAME}' not found directly")
            
        # Test with team context
        user_with_team = linear_client.getUserByName(ASSIGNEE_NAME, TEAM_KEY)
        if user_with_team:
            logger.info(f"Found user with team context: {user_with_team.get('displayName')} with ID: {user_with_team.get('id')}")
        else:
            logger.warning(f"User '{ASSIGNEE_NAME}' not found with team context")
            
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

        # Let's test to directly create the issue using linear_tools
        # This will use the adapter mechanism
        try:
            issue = linear_tools.createIssue(**params)
            if issue:
                issue_number = issue.get("number")
                logger.info(f"Successfully created issue #{issue_number}: {issue.get('title')}")
                logger.info(f"Issue details: {json.dumps(issue, indent=2) if issue else 'None'}")
                
                # Check if assignee was set correctly
                if issue.get("assignee"):
                    logger.info(f"Assignee set correctly to: {issue.get('assignee').get('displayName')}")
                else:
                    logger.warning("Assignee not set in the created issue")
                
                return issue_number
            else:
                logger.error("Failed to create issue: No result returned")
                return None
        except Exception as e:
            logger.error(f"Error using linear_tools.createIssue: {e}")
            # Fall back to manual method for debugging
            logger.info("Falling back to manual method for debugging")
        
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
    """Test updating an issue using adapter-friendly parameters with the actual adapter mechanism"""
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
        
        # Show how parameters would be adapted by the adapter method
        adapted = LinearParameterAdapter.adapt_update_issue(params)
        logger.info(f"Parameters would be adapted to: {json.dumps(adapted)}")
        
        # Now properly use the adapter mechanism through linear_tools
        # This will automatically call LinearParameterAdapter.adapt_update_issue
        # through the @adapt_parameters decorator
        result = linear_tools.updateIssue(**params)
        
        if not result:
            logger.error("Failed to update issue: No result returned")
            return
        
        logger.info(f"Successfully updated issue #{issue_number}: {result.get('title')}")
        logger.info(f"Updated issue details: {json.dumps(result, indent=2)}")
    
    except Exception as e:
        logger.error(f"Error updating issue: {e}")

def test_create_comment_with_adapter(issue_number):
    """Test creating a comment on an issue using adapter-friendly parameters"""
    if not issue_number:
        logger.error("No issue number provided for comment test")
        return False
    
    try:
        # Prepare comment parameters
        timestamp = int(time.time())
        params = {
            "issueNumber": issue_number,
            "teamKey": TEAM_KEY,  # Add team key to properly resolve issue ID
            "commentData": {
                "body": f"This is a test comment created at timestamp {timestamp}.\n\nIt includes:\n- Bullet points\n- Code blocks\n\n```python\nprint('Hello from Linear!')\n```"
            }
        }
        
        # Log the parameters
        logger.info(f"Creating comment with parameters: {json.dumps(params)}")
        
        # Create the comment using linear_tools
        result = linear_tools.createComment(**params)
        
        if not result:
            logger.error("Failed to create comment: No result returned")
            return False
        
        # Log success info
        comment_id = result.get("id")
        comment_body = result.get("body", "")
        logger.info(f"Successfully created comment ID {comment_id}")
        logger.info(f"Comment body: {comment_body[:100]}...")
        logger.info(f"Comment details: {json.dumps(result, indent=2) if result else 'None'}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating comment: {e}")
        return False

def run_tests():
    """Run all tests"""
    logger.info("Starting Linear action tools adapter tests")
    
    # Test creating an issue with adapter
    issue_number = test_create_issue_with_adapter()
    
    if issue_number:
        # Test updating the created issue with adapter
        test_update_issue_with_adapter(issue_number)
        
        # Test creating a comment on the issue
        comment_result = test_create_comment_with_adapter(issue_number)
        if comment_result:
            logger.info("Comment creation test completed successfully")
        else:
            logger.error("Comment creation test failed")
    
    logger.info("Linear action tools adapter tests completed")

def test_assignee_name_handling():
    """Test specifically how the adapter handles assignee_name"""
    logger.info("Testing assignee name handling in adapter")
    
    try:
        # Try getting user info directly
        logger.info(f"Looking up user with display name '{ASSIGNEE_NAME}'")
        
        # First approach: getCurrentUser
        try:
            user_info = linear_tools.getCurrentUser(f"@{ASSIGNEE_NAME}")
            logger.info(f"getCurrentUser result for '@{ASSIGNEE_NAME}': {json.dumps(user_info, indent=2)}")
        except Exception as e:
            logger.error(f"Error in getCurrentUser: {e}")
        
        # Second approach: scan all teams for the user
        teams = ["OPS", "ENG", "RES", "AI", "MKT", "PRO"]
        users = []
        
        for team_key in teams:
            try:
                logger.info(f"Getting users for team {team_key}")
                team_users = linear_tools.getAllUsers(team_key)
                logger.info(f"Found {len(team_users)} users in team {team_key}")
                users.extend(team_users)
            except Exception as e:
                logger.error(f"Error getting users for team {team_key}: {e}")
        
        # Look for matching display name
        found_users = [u for u in users if ASSIGNEE_NAME.lower() in u.get("displayName", "").lower()]
        
        if found_users:
            logger.info(f"Found {len(found_users)} users matching '{ASSIGNEE_NAME}':")
            for user in found_users:
                logger.info(f"- {user.get('displayName')} (ID: {user.get('id')})")
        else:
            logger.warning(f"No users found with display name containing '{ASSIGNEE_NAME}'")
            logger.info("Available user display names:")
            display_names = sorted(list(set([u.get("displayName") for u in users if u.get("displayName")])))
            for name in display_names[:20]:  # Show first 20 to avoid flooding logs
                logger.info(f"- {name}")
            if len(display_names) > 20:
                logger.info(f"... and {len(display_names) - 20} more")
        
        # Test the adapter method directly
        params = {"assignee_name": ASSIGNEE_NAME, "teamKey": TEAM_KEY}
        adapted = LinearParameterAdapter.adapt_create_issue(params)
        logger.info(f"adapt_create_issue result for assignee_name='{ASSIGNEE_NAME}': {json.dumps(adapted)}")
        
        # Try with full name if available
        if found_users:
            full_name = found_users[0].get("displayName")
            params = {"assignee_name": full_name, "teamKey": TEAM_KEY}
            adapted = LinearParameterAdapter.adapt_create_issue(params)
            logger.info(f"adapt_create_issue result for assignee_name='{full_name}': {json.dumps(adapted)}")
    
    except Exception as e:
        logger.error(f"Error in test_assignee_name_handling: {e}")

if __name__ == "__main__":
    # Add assignee name handling test
    test_assignee_name_handling()
    
    # Run the main tests
    run_tests() 