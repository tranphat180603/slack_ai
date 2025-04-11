#!/usr/bin/env python
"""
Automated test file for Linear API methods in tools_declaration.py
Tests filterIssues, createIssue (optional), and updateIssue (optional)
"""

import os
import sys
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the LinearTools class from tools_declaration
from tools.tools_declaration import LinearTools, LinearParameterAdapter

# Set to True to run create/update tests (Warning: creates real issues in Linear)
RUN_CREATE_UPDATE_TESTS = True

def test_filter_issues():
    """Test the filterIssues method with various parameters"""
    print("\n=== Testing filterIssues ===")
    
    # Initialize LinearTools
    linear_tools = LinearTools()
    
    # Test basic filtering by team
    print("\n--- Test 1: Basic Team Filter ---")
    issues = linear_tools.filterIssues(teamKey="OPS", limit=3)
    print(f"Found {len(issues)} issues in OPS team")
    if issues:
        print("First issue:")
        pprint({k: v for k, v in issues[0].items() if k in ['id', 'number', 'title', 'state', 'priority', 'assignee']})
    
    # Test filtering with assignee name
    print("\n--- Test 2: Filter by Assignee ---")
    issues = linear_tools.filterIssues(teamKey="OPS", assignee_name="phat", limit=3)
    print(f"Found {len(issues)} issues assigned to 'phat' in OPS team")
    if issues:
        print("First issue:")
        pprint({k: v for k, v in issues[0].items() if k in ['id', 'number', 'title', 'state', 'priority', 'assignee']})
    
    # Test filtering with state
    print("\n--- Test 3: Filter by State ---")
    issues = linear_tools.filterIssues(teamKey="OPS", state="Todo", limit=3)
    print(f"Found {len(issues)} issues in 'Todo' state in OPS team")
    if issues:
        print("First issue:")
        pprint({k: v for k, v in issues[0].items() if k in ['id', 'number', 'title', 'state', 'priority', 'assignee']})
    
    # Test filtering with priority
    print("\n--- Test 4: Filter by Priority ---")
    issues = linear_tools.filterIssues(teamKey="OPS", priority=3, limit=3)  # Medium priority
    print(f"Found {len(issues)} issues with priority 3 in OPS team")
    if issues:
        print("First issue:")
        pprint({k: v for k, v in issues[0].items() if k in ['id', 'number', 'title', 'state', 'priority', 'assignee']})
    
    # Test filtering with multiple criteria
    print("\n--- Test 5: Filter with Multiple Criteria ---")
    issues = linear_tools.filterIssues(
        teamKey="OPS",
        state="Todo",
        priority=3,
        assignee_name="phat",
        limit=3
    )
    print(f"Found {len(issues)} issues matching multiple criteria")
    if issues:
        print("First issue:")
        pprint({k: v for k, v in issues[0].items() if k in ['id', 'number', 'title', 'state', 'priority', 'assignee']})
    
    # Test with empty/zero values to verify they're properly handled
    print("\n--- Test 6: Handling Empty/Zero Values ---")
    issues = linear_tools.filterIssues(
        teamKey="OPS", 
        priority=0,  # Should be excluded
        assignee_name="",  # Should be excluded
        limit=3
    )
    print(f"Found {len(issues)} issues with empty/zero values properly handled")
    
    # Print how parameters were adapted
    print("\nParameter adaptation example:")
    adapted = LinearParameterAdapter.adapt_filter_issues({
        'teamKey': 'OPS', 
        'priority': 0,  # Should be excluded
        'assignee_name': '',  # Should be excluded
    })
    print("Original params: {'teamKey': 'OPS', 'priority': 0, 'assignee_name': ''}")
    print(f"Adapted params: {adapted}")

def test_create_issue():
    """Test the createIssue method with direct and lookup parameters"""
    print("\n=== Testing createIssue ===")
    
    # Initialize LinearTools
    linear_tools = LinearTools()
    
    # Test creating issue with direct parameters
    print("\n--- Test 1: Create Issue with Direct Parameters ---")
    try:
        # Create a test issue with minimal required fields
        issue = linear_tools.createIssue(
            teamKey="OPS",  # Should lookup teamId
            title="[TEST] API Test Issue - Please Ignore",
            description="This is a test issue created by the Linear API test script.",
            state_name="Todo",  # Should lookup stateId
            priority=3,  # Medium priority
            assignee_name="phat"  # Should lookup assigneeId
        )
        print("Issue created successfully!")
        print(f"Issue ID: {issue.get('id')}")
        print(f"Issue Number: {issue.get('number')}")
        print(f"Title: {issue.get('title')}")
        
        # Print the adapter conversion for debugging
        print("\nParameter adaptation example:")
        adapted = LinearParameterAdapter.adapt_create_issue({
            'teamKey': 'OPS',
            'title': '[TEST] API Test Issue - Please Ignore',
            'description': 'This is a test issue created by the Linear API test script.',
            'state_name': 'Todo',
            'priority': 3,
            'assignee_name': 'phat'
        })
        print(f"Adapted params: {adapted}")
        
        # Return issue number for update test
        return issue.get('number')
    except Exception as e:
        print(f"Error creating issue: {str(e)}")
        return None

def test_update_issue(issue_number):
    """Test the updateIssue method with an existing issue"""
    print(f"\n=== Testing updateIssue for Issue #{issue_number} ===")
    
    # Initialize LinearTools
    linear_tools = LinearTools()
    
    try:
        # First get the issue ID
        print(f"Fetching issue details for issue #{issue_number}...")
        issue_criteria = {"number": {"eq": issue_number}}
        issues = linear_tools.filterIssues(issue_criteria)
        
        if not issues or len(issues) == 0:
            print(f"Error: Could not find issue #{issue_number}")
            return
            
        issue = issues[0]
        issue_id = issue.get('id')
        print(f"Found issue ID: {issue_id}")
        
        # Update the issue
        print("Updating issue...")
        updated_issue = linear_tools.updateIssue(
            issue_id=issue_id,  # Use issue_id instead of issueNumber
            title=f"[TEST UPDATED] API Test Issue #{issue_number}",
            priority=2,  # High priority
            state_name="In Progress"  # Should lookup stateId
        )
        print("Issue updated successfully!")
        print(f"Issue ID: {updated_issue.get('id')}")
        print(f"Issue Number: {updated_issue.get('number')}")
        print(f"Updated Title: {updated_issue.get('title')}")
        print(f"Updated Priority: {updated_issue.get('priority')}")
        state = updated_issue.get('state', {})
        print(f"Updated State: {state.get('name') if state else 'Unknown'}")
        
        # Print the adapter conversion for debugging
        print("\nParameter adaptation example:")
        adapted = LinearParameterAdapter.adapt_update_issue({
            'issue_id': issue_id,  # Use issue_id parameter instead
            'title': f'[TEST UPDATED] API Test Issue #{issue_number}',
            'priority': 2,
            'state_name': 'In Progress'
        })
        print(f"Adapted params: {adapted}")
    except Exception as e:
        print(f"Error updating issue: {str(e)}")

def main():
    """Main function to run all tests"""
    # Test filterIssues (always run)
    test_filter_issues()
    
    # Test create and update only if enabled
    if RUN_CREATE_UPDATE_TESTS:
        print("\n***WARNING: About to create real issues in Linear***")
        print("Running create/update tests because RUN_CREATE_UPDATE_TESTS=True")
        
        # Test createIssue
        issue_number = test_create_issue()
        
        # Test updateIssue with the newly created issue
        if issue_number:
            test_update_issue(issue_number)
        else:
            print("Skipping update test since issue creation failed")
    else:
        print("\nSkipping create/update tests (RUN_CREATE_UPDATE_TESTS=False)")
        print("To enable, set RUN_CREATE_UPDATE_TESTS=True at the top of this file")

if __name__ == "__main__":
    main() 