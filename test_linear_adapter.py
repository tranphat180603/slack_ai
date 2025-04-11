#!/usr/bin/env python3
"""
Comprehensive test script for Linear API adapters.
This script demonstrates how to use the adapter classes in tools_declaration.py
to create and update issues with various parameters.
"""

import os
import logging
import time
import json
from dotenv import load_dotenv
from tools.tools_declaration import LinearTools, LinearParameterAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Test constants
TEAM_KEY = "OPS"
ASSIGNEE_NAME = "phat"  # Use a valid assignee name in your team
CYCLE_NUMBER = 42  # Use a valid cycle number in your team

def test_create_simple_issue():
    """Test creating a simple issue with just the required fields"""
    logger.info("\n=== Testing Simple Issue Creation ===")
    
    # Create LinearTools instance
    linear_tools = LinearTools()
    
    # Create basic issue with minimal required fields
    timestamp = int(time.time())
    params = {
        "teamKey": TEAM_KEY,
        "title": f"Simple Test Issue {timestamp}"
    }
    
    # Show how parameters would be adapted
    logger.info(f"Original parameters: {json.dumps(params)}")
    adapted = LinearParameterAdapter.adapt_create_issue(params)
    logger.info(f"Adapted parameters: {json.dumps(adapted)}")
    
    try:
        # Create the issue
        # Note: Here we're manually passing to the LinearClient what the adapter would do
        result = linear_tools.client.createIssue(adapted.get("data"))
        
        if not result.get("success"):
            logger.error(f"Failed to create issue: {result}")
            return None
        
        issue = result.get("issue", {})
        issue_number = issue.get("number")
        logger.info(f"Successfully created issue #{issue_number}: {issue.get('title')}")
        
        return issue_number
    except Exception as e:
        logger.error(f"Error creating issue: {e}")
        return None

def test_create_complex_issue():
    """Test creating a complex issue with multiple fields"""
    logger.info("\n=== Testing Complex Issue Creation ===")
    
    # Create LinearTools instance
    linear_tools = LinearTools()
    
    # First, get team ID for the team key
    try:
        # Get issues for the team to find team ID
        criteria = {"team": {"key": {"eq": TEAM_KEY}}}
        issues = linear_tools.client.filterIssues(criteria, 1)
        
        if not issues:
            logger.error(f"No issues found for team {TEAM_KEY}")
            return None
        
        team_id = issues[0].get("team", {}).get("id")
        team_name = issues[0].get("team", {}).get("name", "Unknown")
        
        if not team_id:
            logger.error(f"Could not find team ID for team {TEAM_KEY}")
            return None
        
        logger.info(f"Found team: {team_name} (ID: {team_id}, Key: {TEAM_KEY})")
        
        # Create issue with more fields (but we need to look up IDs manually)
        timestamp = int(time.time())
        
        # This would be the ideal parameter format for users/AI
        ideal_params = {
            "teamKey": TEAM_KEY,
            "title": f"Complex Test Issue {timestamp}",
            "description": "This is a test issue created with multiple fields.\n\n"
                          "## Features\n\n- Feature 1\n- Feature 2\n\n"
                          "## Requirements\n\n- Requirement A\n- Requirement B",
            "priority": 3.0,  # Medium priority
            "estimate": 2.0,  # 2 points
            "assignee_name": ASSIGNEE_NAME
            # "cycle_number": CYCLE_NUMBER
        }
        
        logger.info(f"Ideal parameters: {json.dumps(ideal_params)}")
        
        # But in reality we need to look up these IDs manually
        # Get assignee ID
        users = []
        for team in ["OPS", "ENG", "RES", "AI", "MKT", "PRO"]:
            try:
                team_users = linear_tools.getAllUsers(team)
                users.extend(team_users)
            except Exception as e:
                logger.warning(f"Error getting users for team {team}: {e}")
        
        # Find the user with the matching display name
        assignee_id = None
        for user in users:
            if user.get("displayName", "").lower() == ASSIGNEE_NAME.lower():
                assignee_id = user.get("id")
                logger.info(f"Found user ID: {assignee_id} for display name: {ASSIGNEE_NAME}")
                break
        
        if not assignee_id:
            logger.warning(f"No user found with display name: {ASSIGNEE_NAME}")
        
        # Create issue data
        issue_data = {
            "teamId": team_id,
            "title": ideal_params["title"],
            "description": ideal_params["description"],
            "priority": ideal_params["priority"],
            "estimate": ideal_params["estimate"]
        }
        
        # Only add assignee if found
        if assignee_id:
            issue_data["assigneeId"] = assignee_id
        
        # Create the issue
        logger.info(f"Actual request data: {json.dumps(issue_data)}")
        result = linear_tools.client.createIssue(issue_data)
        
        if not result.get("success"):
            logger.error(f"Failed to create issue: {result}")
            return None
        
        issue = result.get("issue", {})
        issue_number = issue.get("number")
        logger.info(f"Successfully created issue #{issue_number}: {issue.get('title')}")
        logger.info(f"Created issue details: {json.dumps(issue, indent=2)}")
        
        return issue_number
        
    except Exception as e:
        logger.error(f"Error creating complex issue: {e}")
        return None

def test_update_issue(issue_number):
    """Test updating an issue"""
    if not issue_number:
        logger.error("No issue number provided for update test")
        return False
    
    logger.info(f"\n=== Testing Issue Update (#{issue_number}) ===")
    
    # Create LinearTools instance
    linear_tools = LinearTools()
    
    try:
        # This would be the ideal parameter format for users/AI
        timestamp = int(time.time())
        params = {
            "issue_number": issue_number,
            "title": f"Updated Test Issue {timestamp}",
            "description": "This issue has been updated by the test script.\n\n"
                          "## Updated Features\n\n- Feature A\n- Feature B\n\n"
                          "## Updated Requirements\n\n- Requirement X\n- Requirement Y",
            "priority": 2.0  # High priority
        }
        
        # Show how parameters would be adapted
        logger.info(f"Original update parameters: {json.dumps(params)}")
        adapted = LinearParameterAdapter.adapt_update_issue(params)
        logger.info(f"Adapted update parameters: {json.dumps(adapted)}")
        
        # For actually updating, we need to find the issue ID first
        criteria = {"team": {"key": {"eq": TEAM_KEY}}}
        issues = linear_tools.client.filterIssues(criteria)
        
        # Find our issue by number
        target_issue = None
        for issue in issues:
            if issue.get("number") == issue_number:
                target_issue = issue
                break
        
        if not target_issue:
            logger.error(f"Could not find issue #{issue_number} in team {TEAM_KEY}")
            return False
        
        # Extract the issue ID
        issue_id = target_issue.get("id")
        
        if not issue_id:
            logger.error(f"Issue #{issue_number} does not have an ID")
            return False
        
        logger.info(f"Found issue ID: {issue_id} for issue #{issue_number}")
        
        # Update the issue using the GraphQL API
        from ops_linear_db.linear_client import gql
        query = gql("""
        mutation issueUpdate($issueId: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $issueId, input: $input) {
            success
            issue {
              id
              number
              title
              description
              priority
              estimate
              state {
                id
                name
              }
              assignee {
                id
                displayName
              }
            }
          }
        }
        """)
        
        # Prepare the update data
        update_data = {
            "title": params["title"],
            "description": params["description"],
            "priority": params["priority"]
        }
        
        variables = {"issueId": issue_id, "input": update_data}
        result = linear_tools.client._execute_query(query, variables)
        update_result = result.get("issueUpdate", {})
        
        if not update_result.get("success"):
            logger.error(f"Failed to update issue: {result}")
            return False
        
        updated_issue = update_result.get("issue", {})
        logger.info(f"Successfully updated issue #{issue_number}: {updated_issue.get('title')}")
        logger.info(f"Updated issue details: {json.dumps(updated_issue, indent=2)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error updating issue: {e}")
        return False

def run_tests():
    """Run comprehensive tests for Linear API adapters"""
    logger.info("Starting comprehensive Linear API adapter tests")
    
    # Test simple issue creation
    simple_issue_number = test_create_simple_issue()
    
    # Test complex issue creation
    complex_issue_number = test_create_complex_issue()
    
    # Test updating the complex issue if created
    if complex_issue_number:
        test_update_issue(complex_issue_number)
    
    logger.info("Comprehensive Linear API adapter tests completed")

if __name__ == "__main__":
    run_tests() 