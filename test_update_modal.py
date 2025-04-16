#!/usr/bin/env python3
"""
Test script for the open_update_issue_modal method in SlackModals
"""

import asyncio
import json
import os
import unittest
from unittest.mock import MagicMock, patch

# Import the SlackModals class
from ops_slack.slack_modals import SlackModals

class MockResponse:
    """Mock response for Slack API calls"""
    def __init__(self, ok=True, view_id="test_view_id"):
        self.data = {"ok": ok, "view": {"id": view_id}}

    def get(self, key, default=None):
        return self.data.get(key, default)

class TestUpdateModal(unittest.TestCase):
    """Test cases for the open_update_issue_modal method"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock Slack client
        self.mock_slack_client = MagicMock()
        self.mock_slack_client.views_open.return_value = MockResponse()
        
        # Create an instance of SlackModals with the mock client
        self.slack_modals = SlackModals(self.mock_slack_client)
        
        # Add a mock linear_tools object to SlackModals instance
        self.slack_modals.linear_tools = MagicMock()
        
        # Mock getAllStates to return some states
        self.slack_modals.linear_tools.getAllStates.return_value = [
            {"name": "Todo"},
            {"name": "In Progress"},
            {"name": "Done"}
        ]
        
        # Mock filterIssues to return a mock issue
        self.slack_modals.linear_tools.filterIssues.return_value = [{
            "id": "test-id",
            "number": 2568,
            "title": "Test Issue",
            "description": "Test Description",
            "priority": 0,
            "team": {"key": "OPS", "name": "Operations"},
            "state": {"name": "Todo"},
            "assignee": {"displayName": "phat"}
        }]
        
        # Mock getAllUsers to return some users
        self.slack_modals.linear_tools.getAllUsers.return_value = [
            {"displayName": "phat"},
            {"displayName": "user2"}
        ]

    async def test_open_update_issue_modal(self):
        """Test open_update_issue_modal method"""
        # Test parameters
        trigger_id = "test_trigger_id"
        issue_number = 2568
        prefilled_data = {
            "team_key": "OPS",
            "title": "Optimize Asynchronous Data Processing Pipelines for Enhanced Throughput and Fault Tolerance",
            "description": "## Context\n\nTest description",
            "priority": 0,
            "assignee_name": "phat",
            "state_name": "Todo",
            "cycle_number": 43
        }
        conversation_id = "test_conversation_id"
        
        # Call the method
        result = await self.slack_modals.open_update_issue_modal(
            trigger_id=trigger_id,
            issue_number=issue_number,
            prefilled_data=prefilled_data,
            conversation_id=conversation_id
        )
        
        # Assert that views_open was called
        self.mock_slack_client.views_open.assert_called_once()
        
        # Check that the view was opened successfully
        self.assertTrue(result)
        
        # Extract the view passed to views_open
        call_args = self.mock_slack_client.views_open.call_args
        view = call_args[1].get("view", {})
        
        # Check basic view structure
        self.assertEqual(view.get("callback_id"), "linear_update_issue_modal")
        
        # Check that the title contains the issue number
        self.assertTrue(str(issue_number) in view.get("title", {}).get("text", ""))
        
        # Check that blocks exist
        self.assertTrue(len(view.get("blocks", [])) > 0)
        
        # Check if priority block is set correctly
        priority_block = None
        for block in view.get("blocks", []):
            if block.get("block_id") == "priority_block":
                priority_block = block
                break
        
        self.assertIsNotNone(priority_block)
        
        # Check if assignee block is set correctly
        assignee_block = None
        for block in view.get("blocks", []):
            if block.get("block_id") == "assignee_block":
                assignee_block = block
                break
        
        self.assertIsNotNone(assignee_block)
        
        # Print the metadata (which should contain team_key and issue_number)
        metadata = json.loads(view.get("private_metadata", "{}"))
        self.assertEqual(metadata.get("issue_number"), issue_number)
        self.assertEqual(metadata.get("team_key"), "OPS")

    async def test_open_update_issue_modal_error(self):
        """Test open_update_issue_modal with error from Slack API"""
        # Make the client raise an exception
        self.mock_slack_client.views_open.side_effect = Exception("Test error")
        
        # Test parameters
        trigger_id = "test_trigger_id"
        issue_number = 2568
        prefilled_data = {
            "team_key": "OPS",
            "title": "Test Issue",
            "description": "Test Description",
            "priority": 0,
            "assignee_name": "phat",
            "state_name": "Todo"
        }
        
        # Call the method
        result = await self.slack_modals.open_update_issue_modal(
            trigger_id=trigger_id,
            issue_number=issue_number,
            prefilled_data=prefilled_data
        )
        
        # Assert that views_open was called
        self.mock_slack_client.views_open.assert_called_once()
        
        # Check that the method returned False due to the error
        self.assertFalse(result)

def run_tests():
    """Run the tests"""
    # Create a test case instance
    test_case = TestUpdateModal()
    
    # Set up the test case
    test_case.setUp()
    
    # Get the asyncio loop
    loop = asyncio.get_event_loop()
    
    # Run the test methods
    loop.run_until_complete(test_case.test_open_update_issue_modal())
    
    # Reset the mock for the second test
    test_case.mock_slack_client.views_open.reset_mock()
    test_case.mock_slack_client.views_open.side_effect = Exception("Test error")
    
    # Run the error test
    loop.run_until_complete(test_case.test_open_update_issue_modal_error())
    
    print("All tests completed successfully!")

if __name__ == "__main__":
    run_tests() 