#!/usr/bin/env python
"""
Test file for testing Linear API methods in tools_declaration.py
Tests for filterIssues, createIssue, and updateIssue
"""

import os
import sys
from pprint import pprint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the LinearTools class from tools_declaration
from tools.tools_declaration import LinearTools, LinearParameterAdapter

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
    print("This will attempt to create a test issue. Proceed? (y/n)")
    response = input().lower()
    
    if response == 'y':
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
            
            # Return issue number for update test
            return issue.get('number')
        except Exception as e:
            print(f"Error creating issue: {str(e)}")
            return None
    else:
        print("Skipping issue creation test")
        return None

def test_update_issue(issue_number=None):
    """Test the updateIssue method with an existing issue"""
    print("\n=== Testing updateIssue ===")
    
    if issue_number is None:
        # Ask for an issue number to update
        print("Enter an issue number to update (or press Enter to skip): ")
        user_input = input().strip()
        if user_input:
            try:
                issue_number = int(user_input)
            except ValueError:
                print("Invalid issue number")
                return
        else:
            print("Skipping issue update test")
            return
    
    # Initialize LinearTools
    linear_tools = LinearTools()
    
    print(f"\n--- Test 1: Update Issue #{issue_number} ---")
    print(f"This will attempt to update issue #{issue_number}. Proceed? (y/n)")
    response = input().lower()
    
    if response == 'y':
        try:
            # Update the issue
            updated_issue = linear_tools.updateIssue(
                issueNumber=issue_number,
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
        except Exception as e:
            print(f"Error updating issue: {str(e)}")
    else:
        print("Skipping issue update test")

def main():
    """Main function to run all tests"""
    # Test filterIssues
    test_filter_issues()
    
    # Test createIssue
    issue_number = test_create_issue()
    
    # Test updateIssue with the newly created issue
    if issue_number:
        test_update_issue(issue_number)
    else:
        test_update_issue()

if __name__ == "__main__":
    main() 