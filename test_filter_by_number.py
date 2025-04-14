#!/usr/bin/env python3
"""
Test script to verify filtering issues by number in Linear API.
"""

import os
import json
import logging
from dotenv import load_dotenv
from ops_linear_db.linear_client import LinearClient
from tools.tools_declaration import LinearParameterAdapter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_filter_by_number():
    """Test filtering an issue by its number"""
    try:
        # Create Linear client
        linear_client = LinearClient(os.getenv("LINEAR_API_KEY"))
        
        # 1. Test with a team key and issue number
        issue_number = 2546  # Use a recent issue number from your previous test
        team_key = "MKT"  # From our previous test, we saw issue 2546 exists in MKT
        
        # Create filter criteria
        criteria = {
            "team": {"key": {"eq": team_key}},
            "number": {"eq": issue_number}
        }
        
        logger.info(f"Filtering for issue #{issue_number} in team {team_key}")
        result = linear_client.filterIssues(criteria, limit=1)
        
        if result and len(result) > 0:
            issue = result[0]
            logger.info(f"Found issue #{issue.get('number')}: {issue.get('title')}")
            logger.info(f"Issue ID: {issue.get('id')}")
            logger.info(f"Team: {issue.get('team', {}).get('key')}")
        else:
            logger.warning(f"No issue found with number {issue_number} in team {team_key}")
            
        # 2. Test with just issue number (no team key)
        criteria = {
            "number": {"eq": issue_number}
        }
        
        logger.info(f"Filtering for issue #{issue_number} without team key")
        result = linear_client.filterIssues(criteria, limit=5)
        
        if result and len(result) > 0:
            logger.info(f"Found {len(result)} issues with number {issue_number}:")
            for issue in result:
                logger.info(f"  #{issue.get('number')} ({issue.get('team', {}).get('key')}): {issue.get('title')}")
                logger.info(f"  Issue ID: {issue.get('id')}")
        else:
            logger.warning(f"No issues found with number {issue_number}")
            
        # 3. Test the adapter function
        logger.info("\nTesting adapter function with issue_number")
        
        # Test with just issue number
        params = {
            "issue_number": issue_number
        }
        adapted = LinearParameterAdapter.adapt_filter_issues(params)
        logger.info(f"Adapter result for issue_number={issue_number}:")
        logger.info(json.dumps(adapted, indent=2))
        
        # Test with issue number and team key
        params = {
            "issue_number": issue_number,
            "team_key": team_key
        }
        adapted = LinearParameterAdapter.adapt_filter_issues(params)
        logger.info(f"Adapter result for issue_number={issue_number}, team_key={team_key}:")
        logger.info(json.dumps(adapted, indent=2))
            
        return True
            
    except Exception as e:
        logger.error(f"Error filtering issue by number: {e}")
        return False

if __name__ == "__main__":
    test_filter_by_number() 