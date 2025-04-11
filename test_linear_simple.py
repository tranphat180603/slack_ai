#!/usr/bin/env python3
"""
Simplified test script for Linear API issue creation and update.
This script uses the LinearClient directly with minimal parameters.
"""

import os
import logging
import time
import json
from dotenv import load_dotenv
from ops_linear_db.linear_client import LinearClient, LinearError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_team_id_for_key(linear_client, team_key):
    """Get team ID for the specified team key"""
    # The easiest way to get a team ID is to get any issue from the team
    # and extract the team ID from the issue
    try:
        # Use the existing filter issues function
        from ops_linear_db.linear_client import gql
        query = gql("""
        query ($teamKey: String!) {
          teams(filter: {key: {eq: $teamKey}}) {
            nodes {
              id
              key
              name
            }
          }
        }
        """)
        
        result = linear_client._execute_query(query, {"teamKey": team_key})
        teams = result.get("teams", {}).get("nodes", [])
        
        if not teams:
            logger.warning(f"No team found with key: {team_key} using GraphQL query")
            # Fallback method: get issues and extract team ID
            criteria = {"team": {"key": {"eq": team_key}}}
            issues = linear_client.filterIssues(criteria, 1)
            
            if not issues:
                raise ValueError(f"No issues found for team {team_key}")
            
            team_id = issues[0].get("team", {}).get("id")
            team_name = issues[0].get("team", {}).get("name", "Unknown")
            
            if not team_id:
                raise ValueError(f"Could not find team ID for team {team_key}")
            
            logger.info(f"Found team using fallback method: {team_name} (ID: {team_id}, Key: {team_key})")
            return team_id
        
        team = teams[0]
        logger.info(f"Found team: {team['name']} (ID: {team['id']}, Key: {team['key']})")
        return team["id"]
        
    except Exception as e:
        logger.error(f"Error getting team ID: {e}")
        # Last resort fallback - use the existing method from our test script
        criteria = {"team": {"key": {"eq": team_key}}}
        issues = linear_client.filterIssues(criteria, 1)
        
        if not issues:
            raise ValueError(f"No issues found for team {team_key}")
        
        team_id = issues[0].get("team", {}).get("id")
        
        if not team_id:
            raise ValueError(f"Could not find team ID for team {team_key}")
        
        logger.info(f"Found team ID using final fallback: {team_id} for team key: {team_key}")
        return team_id

def create_basic_issue(linear_client, team_key="OPS"):
    """Create a basic issue with just the required fields"""
    try:
        # Get team ID for the team key
        team_id = get_team_id_for_key(linear_client, team_key)
        
        # Create a basic issue with only required fields
        timestamp = int(time.time())
        issue_data = {
            "teamId": team_id,
            "title": f"Basic Test Issue {timestamp}"
        }
        
        logger.info(f"Creating issue with data: {json.dumps(issue_data)}")
        result = linear_client.createIssue(issue_data)
        
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

def update_basic_issue(linear_client, issue_number):
    """Update a basic issue with minimal data"""
    if not issue_number:
        logger.error("No issue number provided for update")
        return False
    
    try:
        # Update with minimal data
        timestamp = int(time.time())
        update_data = {
            "title": f"Updated Basic Test Issue {timestamp}",
            "description": "This is a basic description added during update."
        }
        
        logger.info(f"Updating issue #{issue_number} with data: {json.dumps(update_data)}")
        
        # Let's directly use the approach from the document
        from ops_linear_db.linear_client import gql
        
        # This is the approach used in the LinearClient class
        query = gql("""
        mutation issueUpdate($issueId: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $issueId, input: $input) {
            success
            issue {
              id
              number
              title
              description
            }
          }
        }
        """)
        
        # First, we need to get existing issues from the team and find our issue by number
        # Use the filterIssues function we know works
        criteria = {"team": {"key": {"eq": "OPS"}}}
        issues = linear_client.filterIssues(criteria)
        
        # Find our issue by number
        target_issue = None
        for issue in issues:
            if issue.get("number") == issue_number:
                target_issue = issue
                break
        
        if not target_issue:
            logger.error(f"Could not find issue #{issue_number} in team OPS")
            return False
        
        # Extract the issue ID
        issue_id = target_issue.get("id")
        
        if not issue_id:
            logger.error(f"Issue #{issue_number} does not have an ID")
            return False
        
        logger.info(f"Found issue ID: {issue_id} for issue #{issue_number}")
        
        # Update the issue
        variables = {"issueId": issue_id, "input": update_data}
        result = linear_client._execute_query(query, variables)
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
    """Run simple tests for Linear API issue operations"""
    logger.info("Starting simple Linear API tests")
    
    # Initialize Linear client directly
    linear_client = LinearClient(os.getenv("LINEAR_API_KEY"))
    
    # Test creating a basic issue
    issue_number = create_basic_issue(linear_client)
    
    if issue_number:
        # Test updating the issue
        update_basic_issue(linear_client, issue_number)
    
    logger.info("Simple Linear API tests completed")

if __name__ == "__main__":
    run_tests() 