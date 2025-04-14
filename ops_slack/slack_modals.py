"""
Slack Modals Implementation for TMAI Agent
Provides modal windows for user approval/editing before executing actions

This module implements:
1. Linear issue creation modal
2. Linear issue update modal
3. Modal submission handling
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from ops_linear_db.linear_client import LinearClient

# Configure logger
logger = logging.getLogger("slack_modals")

class SlackModals:
    """
    Handles Slack modal windows for user approval flows
    """
    
    def __init__(self, slack_client: WebClient):
        """
        Initialize modal handler with Slack client
        
        Args:
            slack_client: Initialized Slack WebClient instance
        """
        self.slack_client = slack_client
        self.linear_client = None
        
        # Initialize Linear client if API key is available
        linear_api_key = os.environ.get("LINEAR_API_KEY")
        if linear_api_key:
            from ops_linear_db.linear_client import LinearClient
            self.linear_client = LinearClient(linear_api_key)
            
            # Also initialize linear_tools for methods not available in LinearClient
            try:
                from tools.tools_declaration import linear_tools
                self.linear_tools = linear_tools
            except ImportError:
                logger.warning("Could not import linear_tools from tools_declaration")
                self.linear_tools = None
        
        logger.info("SlackModals initialized")
    
    async def open_create_issue_modal(
        self, 
        trigger_id: str, 
        prefilled_data: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> bool:
        """
        Open a modal for creating a Linear issue with prefilled data
        
        Args:
            trigger_id: Slack trigger ID to show the modal
            prefilled_data: Data to prefill in the modal (from AI agent)
            conversation_id: Optional conversation context ID for tracking
            
        Returns:
            True if modal was opened successfully, False otherwise
        """
        try:
            # Get team options for dropdown
            team_options = []
            if self.linear_client:
                try:
                    # Try to use linear_tools.getAllTeams() which exists in tools_declaration.py
                    if hasattr(self, 'linear_tools') and self.linear_tools:
                        teams = self.linear_tools.getAllTeams() or []
                    else:
                        # Fallback to getting teams from issues
                        teams = []
                        raise ImportError("linear_tools not available")
                except Exception as e:
                    logger.warning(f"Error getting teams: {str(e)}")
                    # Fallback: Try to get teams from other sources
                    try:
                        # Get team info from any issues
                        issues = self.linear_client.filterIssues({}, 10)
                        # Extract unique teams
                        teams_set = {}
                        for issue in issues:
                            if issue.get('team') and issue['team'].get('key') and issue['team'].get('name'):
                                team_key = issue['team'].get('key')
                                if team_key not in teams_set:
                                    teams_set[team_key] = {
                                        'key': team_key,
                                        'name': issue['team'].get('name')
                                    }
                        teams = list(teams_set.values())
                    except Exception as inner_e:
                        logger.warning(f"Error getting teams from issues: {str(inner_e)}")
                        teams = []
                
                for team in teams:
                    if team.get("key"):
                        team_options.append({
                            "text": {
                                "type": "plain_text",
                                "text": f"{team.get('name')} ({team.get('key')})"
                            },
                            "value": team.get("key")
                        })
            
            # If no teams found, add default options
            if not team_options:
                # Default options if Linear client not available
                for team_key in ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]:
                    team_options.append({
                        "text": {
                            "type": "plain_text",
                            "text": team_key
                        },
                        "value": team_key
                    })
            
            # Get available states for the selected team
            state_options = []
            selected_team_key = prefilled_data.get("teamKey")
            if selected_team_key and self.linear_client:
                try:
                    states = self.linear_client.getAllStates(selected_team_key)
                    for state in states:
                        if state.get("name"):
                            state_options.append({
                                "text": {
                                    "type": "plain_text",
                                    "text": state.get("name")
                                },
                                "value": state.get("name")
                            })
                except Exception as e:
                    logger.warning(f"Error getting states for team {selected_team_key}: {str(e)}")
            
            # If no states found, add default options
            if not state_options:
                for state_name in ["Todo", "In Progress", "Done", "Canceled"]:
                    state_options.append({
                        "text": {
                            "type": "plain_text",
                            "text": state_name
                        },
                        "value": state_name
                    })
            
            # Get assignee information if available
            assignee_name = prefilled_data.get("assignee_name", "")
            assignee_display = assignee_name
            
            # Try to get proper display name from Linear
            if assignee_name and self.linear_client:
                try:
                    if not assignee_name.startswith("@"):
                        assignee_name = f"@{assignee_name}"
                    
                    user_info = self.linear_client.getCurrentUser(assignee_name)
                    if user_info and user_info.get("linear_display_name"):
                        assignee_display = user_info.get("linear_display_name")
                except Exception as e:
                    logger.warning(f"Error getting user info for {assignee_name}: {str(e)}")
            
            # Get all users for dropdown
            assignee_options = []
            try:
                if self.linear_client:
                    all_users = self.linear_client.getCurrentUser()  # Get all users
                    
                    # Sort users by their linear_display_name
                    sorted_users = sorted(
                        all_users.items(), 
                        key=lambda x: x[1].get("linear_display_name", "").lower()
                    )
                    
                    # Add each user to options
                    for slack_name, user_data in sorted_users:
                        linear_name = user_data.get("linear_display_name")
                        if linear_name:
                            assignee_options.append({
                                "text": {
                                    "type": "plain_text",
                                    "text": linear_name
                                },
                                "value": linear_name
                            })
            except Exception as e:
                logger.warning(f"Error getting users for dropdown: {str(e)}")
            
            # If no options found, add a placeholder
            if not assignee_options:
                assignee_options.append({
                    "text": {
                        "type": "plain_text",
                        "text": "No assignee"
                    },
                    "value": ""
                })
            
            # Construct modal view
            view = {
                "type": "modal",
                "callback_id": "linear_create_issue_modal",
                "private_metadata": json.dumps({
                    "conversation_id": conversation_id,
                    "action": "create_issue",
                    "prefilled_data": prefilled_data
                }),
                "title": {
                    "type": "plain_text",
                    "text": "Create Linear Issue"
                },
                "submit": {
                    "type": "plain_text",
                    "text": "Create"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Cancel"
                },
                "blocks": [
                    # Team selection
                    {
                        "type": "input",
                        "block_id": "team_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "team_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select a team"
                            },
                            "options": team_options,
                            "initial_option": next(
                                (opt for opt in team_options if opt["value"] == prefilled_data.get("teamKey")), 
                                team_options[0] if team_options else None
                            )
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Team"
                        }
                    },
                    # Title
                    {
                        "type": "input",
                        "block_id": "title_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "title_input",
                            "initial_value": prefilled_data.get("title", ""),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Issue title"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Title"
                        }
                    },
                    # Description
                    {
                        "type": "input",
                        "block_id": "description_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "description_input",
                            "multiline": True,
                            "initial_value": prefilled_data.get("description", ""),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Issue description (supports markdown)"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Description"
                        },
                        "optional": True
                    },
                    # Priority
                    {
                        "type": "input",
                        "block_id": "priority_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "priority_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select priority"
                            },
                            "options": [
                                {
                                    "text": {"type": "plain_text", "text": "No priority"},
                                    "value": "0.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "Urgent"},
                                    "value": "1.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "High"},
                                    "value": "2.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "Medium"},
                                    "value": "3.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "Low"},
                                    "value": "4.0"
                                }
                            ],
                            "initial_option": {
                                "text": {"type": "plain_text", "text": self._get_priority_text(prefilled_data.get("priority", 0.0))},
                                "value": str(prefilled_data.get("priority", 0.0))
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Priority"
                        },
                        "optional": True
                    },
                    # Status (state)
                    {
                        "type": "input",
                        "block_id": "state_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "state_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select status"
                            },
                            "options": state_options,
                            "initial_option": next(
                                (opt for opt in state_options if opt["value"] == prefilled_data.get("state_name", "Todo")),
                                state_options[0] if state_options else None
                            )
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Status"
                        },
                        "optional": True
                    },
                    # Assignee
                    {
                        "type": "input",
                        "block_id": "assignee_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "assignee_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select assignee"
                            },
                            "options": assignee_options,
                            "initial_option": next(
                                (opt for opt in assignee_options if opt["value"] == assignee_display),
                                {"text": {"type": "plain_text", "text": "No assignee"}, "value": ""} if not assignee_display else None
                            )
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Assignee"
                        },
                        "optional": True
                    },
                    # Labels
                    {
                        "type": "input",
                        "block_id": "labels_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "labels_input",
                            "initial_value": self._format_labels(prefilled_data.get("label_names", [])),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Labels (comma separated)"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Labels"
                        },
                        "optional": True
                    },
                    # Project
                    {
                        "type": "input",
                        "block_id": "project_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "project_input",
                            "initial_value": prefilled_data.get("project_name", ""),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Project name"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Project"
                        },
                        "optional": True
                    }
                ]
            }
            
            # Open the modal
            response = self.slack_client.views_open(
                trigger_id=trigger_id,
                view=view
            )
            
            logger.info(f"Modal opened with view ID: {response.get('view', {}).get('id')}")
            return True
            
        except SlackApiError as e:
            logger.error(f"Error opening create issue modal: {e.response['error']}")
            return False
    
    async def open_update_issue_modal(
        self, 
        trigger_id: str, 
        issue_number: int,
        prefilled_data: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> bool:
        """
        Open a modal for updating a Linear issue with prefilled data
        
        Args:
            trigger_id: Slack trigger ID to show the modal
            issue_number: Linear issue number to update
            prefilled_data: Data to prefill in the modal (from AI agent)
            conversation_id: Optional conversation context ID for tracking
            
        Returns:
            True if modal was opened successfully, False otherwise
        """
        try:
            # Try to get current issue data if not provided
            issue_data = prefilled_data
            team_key = None
            
            if self.linear_client and not prefilled_data.get("title"):
                try:
                    issues = self.linear_client.filterIssues({"number": issue_number}, 1) or []
                    if issues:
                        issue = issues[0]
                        team_key = issue.get("team", {}).get("key")
                        issue_data = {
                            "title": issue.get("title", ""),
                            "description": issue.get("description", ""),
                            "priority": issue.get("priority", 0.0),
                            "state_name": issue.get("state", {}).get("name", "Todo"),
                            "assignee_name": issue.get("assignee", {}).get("displayName", ""),
                            "label_names": [label.get("name") for label in issue.get("labels", {}).get("nodes", [])],
                            "project_name": issue.get("project", {}).get("name", ""),
                            "teamKey": team_key
                        }
                except Exception as e:
                    logger.warning(f"Error fetching issue data: {str(e)}")
            else:
                team_key = prefilled_data.get("teamKey")
            
            # Get available states for the team
            state_options = []
            if team_key and self.linear_client:
                try:
                    states = self.linear_client.getAllStates(team_key)
                    for state in states:
                        if state.get("name"):
                            state_options.append({
                                "text": {
                                    "type": "plain_text",
                                    "text": state.get("name")
                                },
                                "value": state.get("name")
                            })
                except Exception as e:
                    logger.warning(f"Error getting states for team {team_key}: {str(e)}")
            
            # If no states found, add default options
            if not state_options:
                for state_name in ["Todo", "In Progress", "Done", "Canceled"]:
                    state_options.append({
                        "text": {
                            "type": "plain_text",
                            "text": state_name
                        },
                        "value": state_name
                    })
            
            # Get assignee information if available
            assignee_name = issue_data.get("assignee_name", "")
            assignee_display = assignee_name
            
            # Try to get proper display name from Linear
            if assignee_name and self.linear_client:
                try:
                    if not assignee_name.startswith("@"):
                        assignee_name = f"@{assignee_name}"
                    
                    user_info = self.linear_client.getCurrentUser(assignee_name)
                    if user_info and user_info.get("linear_display_name"):
                        assignee_display = user_info.get("linear_display_name")
                except Exception as e:
                    logger.warning(f"Error getting user info for {assignee_name}: {str(e)}")
            
            # Get all users for dropdown
            assignee_options = []
            try:
                if self.linear_client:
                    all_users = self.linear_client.getCurrentUser()  # Get all users
                    
                    # Sort users by their linear_display_name
                    sorted_users = sorted(
                        all_users.items(), 
                        key=lambda x: x[1].get("linear_display_name", "").lower()
                    )
                    
                    # Add each user to options
                    for slack_name, user_data in sorted_users:
                        linear_name = user_data.get("linear_display_name")
                        if linear_name:
                            assignee_options.append({
                                "text": {
                                    "type": "plain_text",
                                    "text": linear_name
                                },
                                "value": linear_name
                            })
            except Exception as e:
                logger.warning(f"Error getting users for dropdown: {str(e)}")
            
            # If no options found, add a placeholder
            if not assignee_options:
                assignee_options.append({
                    "text": {
                        "type": "plain_text",
                        "text": "No assignee"
                    },
                    "value": ""
                })
            
            # Construct modal view
            view = {
                "type": "modal",
                "callback_id": "linear_update_issue_modal",
                "private_metadata": json.dumps({
                    "conversation_id": conversation_id,
                    "action": "update_issue",
                    "issue_number": issue_number,
                    "prefilled_data": prefilled_data
                }),
                "title": {
                    "type": "plain_text",
                    "text": f"Update Issue #{issue_number}"
                },
                "submit": {
                    "type": "plain_text",
                    "text": "Update"
                },
                "close": {
                    "type": "plain_text",
                    "text": "Cancel"
                },
                "blocks": [
                    # Issue number (non-editable)
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Updating issue #{issue_number}*"
                        }
                    },
                    # Title
                    {
                        "type": "input",
                        "block_id": "title_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "title_input",
                            "initial_value": issue_data.get("title", ""),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Issue title"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Title"
                        }
                    },
                    # Description
                    {
                        "type": "input",
                        "block_id": "description_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "description_input",
                            "multiline": True,
                            "initial_value": issue_data.get("description", ""),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Issue description (supports markdown)"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Description"
                        },
                        "optional": True
                    },
                    # Priority
                    {
                        "type": "input",
                        "block_id": "priority_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "priority_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select priority"
                            },
                            "options": [
                                {
                                    "text": {"type": "plain_text", "text": "No priority"},
                                    "value": "0.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "Urgent"},
                                    "value": "1.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "High"},
                                    "value": "2.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "Medium"},
                                    "value": "3.0"
                                },
                                {
                                    "text": {"type": "plain_text", "text": "Low"},
                                    "value": "4.0"
                                }
                            ],
                            "initial_option": {
                                "text": {"type": "plain_text", "text": self._get_priority_text(issue_data.get("priority", 0.0))},
                                "value": str(issue_data.get("priority", 0.0))
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Priority"
                        },
                        "optional": True
                    },
                    # Status (state)
                    {
                        "type": "input",
                        "block_id": "state_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "state_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select status"
                            },
                            "options": state_options,
                            "initial_option": next(
                                (opt for opt in state_options if opt["value"] == issue_data.get("state_name", "Todo")),
                                state_options[0] if state_options else None
                            )
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Status"
                        },
                        "optional": True
                    },
                    # Assignee
                    {
                        "type": "input",
                        "block_id": "assignee_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "assignee_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select assignee"
                            },
                            "options": assignee_options,
                            "initial_option": next(
                                (opt for opt in assignee_options if opt["value"] == assignee_display),
                                {"text": {"type": "plain_text", "text": "No assignee"}, "value": ""} if not assignee_display else None
                            )
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Assignee"
                        },
                        "optional": True
                    },
                    # Labels
                    {
                        "type": "input",
                        "block_id": "labels_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "labels_input",
                            "initial_value": self._format_labels(issue_data.get("label_names", [])),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Labels (comma separated)"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Labels"
                        },
                        "optional": True
                    },
                    # Project
                    {
                        "type": "input",
                        "block_id": "project_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "project_input",
                            "initial_value": issue_data.get("project_name", ""),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Project name"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Project"
                        },
                        "optional": True
                    }
                ]
            }
            
            # Open the modal
            response = self.slack_client.views_open(
                trigger_id=trigger_id,
                view=view
            )
            
            logger.info(f"Modal opened with view ID: {response.get('view', {}).get('id')}")
            return True
            
        except SlackApiError as e:
            logger.error(f"Error opening update issue modal: {e.response['error']}")
            return False
    
    async def handle_view_submission(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle submission of modal views
        
        Args:
            payload: The view submission payload from Slack
            
        Returns:
            Result of the action with status and details for app.py to process
            For Slack UI: returns empty dict on success, or error dict on failure
        """
        try:
            # Extract view and metadata
            view = payload.get("view", {})
            view_id = view.get("id", "")
            callback_id = view.get("callback_id", "")
            
            # Parse metadata
            metadata = {}
            try:
                metadata_str = view.get("private_metadata", "{}")
                metadata = json.loads(metadata_str)
            except json.JSONDecodeError:
                logger.error(f"Error parsing view metadata: {view.get('private_metadata')}")
            
            # Extract values from view state
            state_values = view.get("state", {}).get("values", {})
            
            # Determine which action to perform
            result = {}
            if callback_id == "linear_create_issue_modal":
                result = await self._handle_create_issue(state_values, metadata)
            elif callback_id == "linear_update_issue_modal":
                result = await self._handle_update_issue(state_values, metadata)
            else:
                logger.warning(f"Unknown callback_id: {callback_id}")
                result = {
                    "success": False,
                    "message": "Unknown action type",
                    "details": {}
                }
                
            # Return the result for app.py to process
            # This doesn't go directly to Slack, but to the app.py handler
            return result
                
        except Exception as e:
            logger.error(f"Error handling view submission: {str(e)}")
            return {
                "success": False,
                "message": f"Error: {str(e)}",
                "details": {}
            }
    
    async def _handle_create_issue(self, state_values: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Linear issue creation from modal submission
        
        Args:
            state_values: Values from the modal
            metadata: Metadata from the modal
            
        Returns:
            Result with status and issue details
        """
        try:
            # Get original prefilled data from metadata
            prefilled_data = metadata.get("prefilled_data", {})
            
            # Extract values from modal
            team_key = state_values.get("team_block", {}).get("team_select", {}).get("selected_option", {}).get("value")
            title = state_values.get("title_block", {}).get("title_input", {}).get("value", "")
            description = state_values.get("description_block", {}).get("description_input", {}).get("value", "")
            priority_str = state_values.get("priority_block", {}).get("priority_select", {}).get("selected_option", {}).get("value", "0.0")
            
            # Extract status using the select element now instead of text input
            state_name = state_values.get("state_block", {}).get("state_select", {}).get("selected_option", {}).get("value", "Todo")
            
            # Get assignee from dropdown
            assignee_name = state_values.get("assignee_block", {}).get("assignee_select", {}).get("selected_option", {}).get("value", "")
            
            labels_str = state_values.get("labels_block", {}).get("labels_input", {}).get("value", "")
            project_name = state_values.get("project_block", {}).get("project_input", {}).get("value", "")
            
            # Convert values to appropriate types
            try:
                priority = float(priority_str)
            except ValueError:
                priority = 0.0
            
            # Parse labels
            label_names = []
            if labels_str:
                label_names = [label.strip() for label in labels_str.split(",") if label.strip()]
            
            # Prepare parameters from modal inputs
            create_params = {
                "teamKey": team_key,
                "title": title,
                "description": description,
                "priority": priority,
                "state_name": state_name,
                "assignee_name": assignee_name,
                "label_names": label_names,
                "project_name": project_name
            }
            
            # Merge with prefilled data for parameters not in the modal
            # This preserves parameters like parent_issue_number, cycle_name, etc.
            for key, value in prefilled_data.items():
                if key not in create_params and value is not None:
                    create_params[key] = value
            
            logger.info(f"Creating issue with parameters: {create_params}")
            
            # Execute the action using Linear tools
            try:
                # First check if we have linear_tools already initialized
                if hasattr(self, 'linear_tools') and self.linear_tools:
                    issue = self.linear_tools.createIssue(**create_params)
                else:
                    # Fall back to importing and using tools_declaration
                    from tools.tools_declaration import linear_tools
                    issue = linear_tools.createIssue(**create_params)
                
                # Add detailed logging about the returned issue object
                logger.debug(f"Issue created - returned object type: {type(issue)}, content: {issue}")
                
                if not issue:
                    return {
                        "success": False,
                        "message": "Failed to create issue",
                        "details": {}
                    }
                
                # Ensure we have a valid issue number
                issue_number = issue.get('number') if isinstance(issue, dict) else None
                if not issue_number and isinstance(issue, dict) and 'id' in issue:
                    # Try to extract number from other fields if available
                    if 'identifier' in issue:
                        # Format might be "TEAM-123"
                        try:
                            issue_number = int(issue['identifier'].split('-')[1])
                        except (IndexError, ValueError):
                            pass
                
                logger.info(f"Issue created successfully with number: {issue_number}")
                
                return {
                    "success": True,
                    "message": f"Issue #{issue_number if issue_number else 'unknown'} created successfully",
                    "details": issue
                }
            except Exception as e:
                logger.error(f"Error creating issue: {str(e)}")
                return {
                    "success": False,
                    "message": f"Error creating issue: {str(e)}",
                    "details": {}
                }
            
        except Exception as e:
            logger.error(f"Error creating issue: {str(e)}")
            return {
                "success": False,
                "message": f"Error creating issue: {str(e)}",
                "details": {}
            }
    
    async def _handle_update_issue(self, state_values: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle Linear issue update from modal submission
        
        Args:
            state_values: Values from the modal
            metadata: Metadata from the modal
            
        Returns:
            Result with status and issue details
        """
        try:
            # Get original prefilled data from metadata
            prefilled_data = metadata.get("prefilled_data", {})
            
            # Get issue number from metadata
            issue_number = metadata.get("issue_number")
            if not issue_number:
                return {
                    "success": False,
                    "message": "Issue number not provided in metadata",
                    "details": {}
                }
            
            # Extract values from modal
            title = state_values.get("title_block", {}).get("title_input", {}).get("value", "")
            description = state_values.get("description_block", {}).get("description_input", {}).get("value", "")
            priority_str = state_values.get("priority_block", {}).get("priority_select", {}).get("selected_option", {}).get("value", "0.0")
            state_name = state_values.get("state_block", {}).get("state_select", {}).get("selected_option", {}).get("value", "")
            
            # Get assignee from dropdown
            assignee_name = state_values.get("assignee_block", {}).get("assignee_select", {}).get("selected_option", {}).get("value", "")
            
            labels_str = state_values.get("labels_block", {}).get("labels_input", {}).get("value", "")
            project_name = state_values.get("project_block", {}).get("project_input", {}).get("value", "")
            
            # Convert values to appropriate types
            try:
                priority = float(priority_str)
            except ValueError:
                priority = 0.0
                
            # Parse labels
            label_names = []
            if labels_str:
                label_names = [label.strip() for label in labels_str.split(",") if label.strip()]
            
            # Prepare parameters from modal inputs
            update_params = {
                "issue_number": issue_number,
                "title": title,
                "description": description,
                "priority": priority,
                "state_name": state_name,
                "assignee_name": assignee_name,
                "label_names": label_names,
                "project_name": project_name
            }
            
            # Merge with prefilled data for parameters not in the modal
            # This preserves parameters like cycle_name, etc.
            for key, value in prefilled_data.items():
                if key not in update_params and value is not None and key != "issue_number":
                    update_params[key] = value
            
            logger.info(f"Updating issue #{issue_number} with parameters: {update_params}")
            
            # Execute the action using Linear tools
            try:
                # First check if we have linear_tools already initialized
                if hasattr(self, 'linear_tools') and self.linear_tools:
                    issue = self.linear_tools.updateIssue(**update_params)
                else:
                    # Fall back to importing and using tools_declaration
                    from tools.tools_declaration import linear_tools
                    issue = linear_tools.updateIssue(**update_params)
                
                # Add detailed logging about the returned issue object
                logger.debug(f"Issue updated - returned object type: {type(issue)}, content: {issue}")
                
                if not issue:
                    return {
                        "success": False,
                        "message": f"Failed to update issue #{issue_number}",
                        "details": {}
                    }
                
                # Verify we have a proper issue object
                is_valid_issue = isinstance(issue, dict) and (issue.get('id') or issue.get('number'))
                
                logger.info(f"Issue #{issue_number} updated successfully")
                
                return {
                    "success": True,
                    "message": f"Issue #{issue_number} updated successfully",
                    "details": issue
                }
            except Exception as e:
                logger.error(f"Error updating issue: {str(e)}")
                return {
                    "success": False,
                    "message": f"Error updating issue: {str(e)}",
                    "details": {}
                }
            
        except Exception as e:
            logger.error(f"Error updating issue: {str(e)}")
            return {
                "success": False,
                "message": f"Error updating issue: {str(e)}",
                "details": {}
            }
    
    def _get_priority_text(self, priority: Union[float, int, str]) -> str:
        """Convert priority value to display text"""
        try:
            priority_float = float(priority)
            priority_map = {
                0.0: "No priority",
                1.0: "Urgent",
                2.0: "High",
                3.0: "Medium",
                4.0: "Low"
            }
            return priority_map.get(priority_float, "No priority")
        except (ValueError, TypeError):
            return "No priority"
    
    def _format_labels(self, labels: List[str]) -> str:
        """Format labels list as comma-separated string"""
        if not labels:
            return ""
        return ", ".join(labels) 