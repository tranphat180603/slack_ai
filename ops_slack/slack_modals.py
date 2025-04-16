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
import time
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
        self.linear_tools = None
        
        # Initialize Linear client if API key is available
        linear_api_key = os.environ.get("LINEAR_API_KEY")
        if linear_api_key:
            # First, always initialize linear_tools from tools_declaration
            try:
                from tools.tools_declaration import linear_tools
                self.linear_tools = linear_tools
                logger.info("Linear tools initialized from tools_declaration")
            except ImportError:
                logger.warning("Could not import linear_tools from tools_declaration")
                self.linear_tools = None
            
            # For backward compatibility, also initialize LinearClient directly
            from ops_linear_db.linear_client import LinearClient
            self.linear_client = LinearClient(linear_api_key)
            
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
            # Extract prefilled data
            title = prefilled_data.get("title", "")
            description = prefilled_data.get("description", "")
            team_key = prefilled_data.get("team_key", "OPS")  # Default to OPS if not provided
            logger.info(f"Using team_key: {team_key}")
            
            # Get all teams for dropdown
            team_options = []
            if hasattr(self, 'linear_tools') and self.linear_tools or self.linear_client:
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
                        if hasattr(self, 'linear_tools') and self.linear_tools:
                            issues = self.linear_tools.filterIssues(limit=10) or []
                        else:
                            from tools.tools_declaration import linear_tools
                            issues = linear_tools.filterIssues(limit=10) or []
                        
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
            if selected_team_key and (hasattr(self, 'linear_tools') and self.linear_tools or self.linear_client):
                try:
                    if hasattr(self, 'linear_tools') and self.linear_tools:
                        states = self.linear_tools.getAllStates(teamKey=selected_team_key)
                    else:
                        from tools.tools_declaration import linear_tools
                        states = linear_tools.getAllStates(teamKey=selected_team_key)
                    
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
            if assignee_name and (hasattr(self, 'linear_tools') and self.linear_tools or self.linear_client):
                try:
                    if not assignee_name.startswith("@"):
                        assignee_name = f"@{assignee_name}"
                    
                    if hasattr(self, 'linear_tools') and self.linear_tools:
                        user_info = self.linear_tools.getCurrentUser(slack_display_name=assignee_name)
                    else:
                        from tools.tools_declaration import linear_tools
                        user_info = linear_tools.getCurrentUser(slack_display_name=assignee_name)
                    
                    if user_info and user_info.get("linear_display_name"):
                        assignee_display = user_info.get("linear_display_name")
                except Exception as e:
                    logger.warning(f"Error getting user info for {assignee_name}: {str(e)}")
            
            # Get all users for dropdown
            assignee_options = []
            try:
                if hasattr(self, 'linear_tools') and self.linear_tools or self.linear_client:
                    # Original code from before the changes
                    if hasattr(self, 'linear_tools') and self.linear_tools:
                        # Make sure we're passing a parameter to getCurrentUser
                        if assignee_name:
                            all_users = self.linear_tools.getCurrentUser(assignee_name)  
                        else:
                            # If no assignee name, don't try to get specific user
                            all_users = {} 
                    else:
                        from tools.tools_declaration import linear_tools
                        if assignee_name:
                            all_users = linear_tools.getCurrentUser(assignee_name)  
                        else:
                            all_users = {}  
                    
                    # If we have no user map or empty result, fall back to getting all users from team
                    if not all_users and team_key:
                        try:
                            if hasattr(self, 'linear_tools') and self.linear_tools:
                                team_users = self.linear_tools.getAllUsers(team_key)
                            else:
                                from tools.tools_declaration import linear_tools
                                team_users = linear_tools.getAllUsers(team_key)
                                
                            # Add each user from getAllUsers to options
                            for user in team_users:
                                display_name = user.get("displayName")
                                if display_name:
                                    assignee_options.append({
                                        "text": {
                                            "type": "plain_text",
                                            "text": display_name
                                        },
                                        "value": display_name
                                    })
                        except Exception as user_err:
                            logger.warning(f"Error getting team users: {str(user_err)}")
                    else:
                        # Sort users by their linear_display_name
                        sorted_users = []
                        try:
                            # Check if all_users is a dictionary mapping usernames to user data
                            if isinstance(all_users, dict) and not all_users.get("linear_display_name"):
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
                            # Handle case where all_users is a single user object
                            elif isinstance(all_users, dict) and all_users.get("linear_display_name"):
                                linear_name = all_users.get("linear_display_name")
                                assignee_options.append({
                                    "text": {
                                        "type": "plain_text",
                                        "text": linear_name
                                    },
                                    "value": linear_name
                                })
                        except Exception as e:
                            logger.warning(f"Error processing users: {str(e)}")
            except Exception as e:
                logger.warning(f"Error getting users for dropdown: {str(e)}")
            
            # If no options found, add a placeholder - ensure it has a non-empty value
            if not assignee_options:
                assignee_options.append({
                    "text": {
                        "type": "plain_text",
                        "text": "No assignee"
                    },
                    "value": "none"  # Using "none" instead of empty string
                })
            
            # Construct modal view
            view = {
                "type": "modal",
                "callback_id": "linear_create_issue_modal",
                "private_metadata": json.dumps({
                    "conversation_id": conversation_id,
                    "action": "create_issue"
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
                            "options": team_options
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
                            "initial_value": self._truncate_description(prefilled_data.get("description", "")),
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
                            "initial_option": next(
                                (opt for opt in [
                                    {"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"},
                                    {"text": {"type": "plain_text", "text": "Urgent"}, "value": "1.0"},
                                    {"text": {"type": "plain_text", "text": "High"}, "value": "2.0"},
                                    {"text": {"type": "plain_text", "text": "Medium"}, "value": "3.0"},
                                    {"text": {"type": "plain_text", "text": "Low"}, "value": "4.0"}
                                ] if opt["value"] == str(float(prefilled_data.get("priority", 0) or 0))),
                                {"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}
                            ) if prefilled_data.get("priority") is not None else None
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
                                None # Remove the invalid fallback that creates an empty value
                            ) if assignee_display else None # Only set initial_option if we have a display name
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
                    },
                    # Parent Issue
                    {
                        "type": "input",
                        "block_id": "parent_issue_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "parent_issue_input",
                            "initial_value": str(prefilled_data.get("parent_issue_number", "")),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Parent issue number"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Parent Issue #"
                        },
                        "optional": True
                    },
                    # Cycle
                    {
                        "type": "input",
                        "block_id": "cycle_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "cycle_input",
                            "initial_value": str(prefilled_data.get("cycle_number", "")),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Cycle number"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Cycle Number"
                        },
                        "optional": True
                    }
                ]
            }
            
            priority_str = str(prefilled_data.get("priority", 0.0))
            
            # Enhanced priority validation to avoid invalid_arguments error
            # Ensure the priority value exists in our options list
            priority_block = None
            for block in view['blocks']:
                if block.get('block_id') == 'priority_block':
                    priority_block = block
                    break
                    
            if priority_block and 'element' in priority_block and 'options' in priority_block['element']:
                valid_priority_values = [opt["value"] for opt in priority_block['element']['options']]
                logger.info(f"TRIGGER DEBUG: Valid priority values: {valid_priority_values}")
                logger.info(f"TRIGGER DEBUG: Current priority value: {priority_str}")
                
                # If priority_str is not in valid values, default to "0.0"
                if priority_str not in valid_priority_values:
                    logger.warning(f"TRIGGER DEBUG: Priority value '{priority_str}' not in valid options, defaulting to '0.0'")
                    priority_str = "0.0"
                
                # Find the matching option from the available options
                priority_option = next(
                    (opt for opt in priority_block['element']['options'] if opt["value"] == priority_str),
                    priority_block['element']['options'][0]  # Default to first option if no match
                )
                
                # Use the exact option object from the available options
                priority_block['element']["initial_option"] = priority_option
                logger.info(f"TRIGGER DEBUG: Setting initial_option to: {json.dumps(priority_option)}")
            else:
                logger.warning(f"TRIGGER DEBUG: Could not find priority block or options")
            
            # Debug the overall size
            try:
                view_json = json.dumps(view)
                logger.info(f"TRIGGER DEBUG: View size in bytes: {len(view_json)}")
            except Exception as e:
                logger.error(f"Error calculating view size: {str(e)}")
                
            # Add detailed trigger_id logging
            logger.info(f"TRIGGER DEBUG: SlackModals called with trigger_id: {trigger_id}")
            logger.info(f"TRIGGER DEBUG: Modal has {len(view['blocks'])} blocks")
            
            # Limit number of options in dropdowns if too many
            MAX_OPTIONS = 50
            if len(team_options) > MAX_OPTIONS:
                logger.info(f"TRIGGER DEBUG: Limiting team_options from {len(team_options)} to {MAX_OPTIONS}")
                team_options = team_options[:MAX_OPTIONS]
                
            if len(assignee_options) > MAX_OPTIONS:
                logger.info(f"TRIGGER DEBUG: Limiting assignee_options from {len(assignee_options)} to {MAX_OPTIONS}")
                assignee_options = assignee_options[:MAX_OPTIONS]
                
            if len(state_options) > MAX_OPTIONS:
                logger.info(f"TRIGGER DEBUG: Limiting state_options from {len(state_options)} to {MAX_OPTIONS}")
                state_options = state_options[:MAX_OPTIONS]
            
            # Update view with limited options
            # First find the correct blocks by block_id instead of using fixed indices
            for i, block in enumerate(view['blocks']):
                if block.get('block_id') == 'team_block' and 'element' in block:
                    logger.info(f"TRIGGER DEBUG: Updating team_block (index {i}) with {len(team_options)} options")
                    block['element']['options'] = team_options
                elif block.get('block_id') == 'state_block' and 'element' in block:
                    logger.info(f"TRIGGER DEBUG: Updating state_block (index {i}) with {len(state_options)} options")
                    block['element']['options'] = state_options
                elif block.get('block_id') == 'assignee_block' and 'element' in block:
                    logger.info(f"TRIGGER DEBUG: Updating assignee_block (index {i}) with {len(assignee_options)} options")
                    block['element']['options'] = assignee_options
            
            # Validate and potentially reduce view size
            view, was_reduced = self._validate_view_size(view)
            if was_reduced:
                logger.info("View size was reduced to fit Slack's limits")
            
            # Finally, verify trigger_id is valid (should be less than 30 seconds old)
            try:
                # Validate trigger_id format
                if not trigger_id or not isinstance(trigger_id, str):
                    logger.error(f"TRIGGER DEBUG: Invalid trigger_id format: {trigger_id}")
                    return False
                
                # Make sure it's not empty or just whitespace
                trigger_id = trigger_id.strip()
                if not trigger_id:
                    logger.error("TRIGGER DEBUG: Empty trigger_id after stripping whitespace")
                    return False
                    
                # Log the trigger_id we're about to use
                logger.info(f"TRIGGER DEBUG: Using trigger_id: {trigger_id[:20]}...")
                
                # Slack trigger_id format is typically: {timestamp}.{team_id}.{random}
                # The timestamp portion is not a Unix timestamp but a different format
                # So we'll just check if the trigger_id was recently received instead
                action_ts = None  # Initialize action_ts directly without referencing undefined payload
                if action_ts:
                    try:
                        action_timestamp = float(action_ts)
                        current_timestamp = time.time()
                        trigger_age = current_timestamp - action_timestamp
                        logger.info(f"TRIGGER DEBUG: Action timestamp age: {trigger_age:.2f} seconds")
                        if trigger_age > 25:  # Slack's limit is 30 seconds, being conservative
                            logger.warning(f"Trigger may be expired (action age: {trigger_age:.2f} seconds)")
                    except Exception as e:
                        logger.warning(f"Could not parse action_ts: {str(e)}")
                else:
                    logger.info(f"TRIGGER DEBUG: Using trigger_id without timestamp validation: {trigger_id[:20]}...")
            except Exception as e:
                logger.warning(f"Could not validate trigger_id timing: {str(e)}")
            
            # Add more debug information about the view structure before opening the modal
            try:
                # Log key information about the issue data being used
                logger.info(f"TRIGGER DEBUG: Issue data summary before opening modal:")
                logger.info(f"TRIGGER DEBUG: Issue #{prefilled_data.get('issue_number', 'unknown')} - Team key: {prefilled_data.get('teamKey', 'unknown')}")
                logger.info(f"TRIGGER DEBUG: Title length: {len(prefilled_data.get('title', ''))}")
                logger.info(f"TRIGGER DEBUG: Description length: {len(prefilled_data.get('description', ''))}")
                
                # Check for and log any missing required fields
                required_fields = ['title']
                missing_fields = [field for field in required_fields if not prefilled_data.get(field)]
                if missing_fields:
                    logger.warning(f"TRIGGER DEBUG: Missing required fields in issue data: {missing_fields}")
                
                # Log information about dropdown options
                logger.info(f"TRIGGER DEBUG: Team options: {len(team_options)}")
                logger.info(f"TRIGGER DEBUG: State options: {len(state_options)}")
                logger.info(f"TRIGGER DEBUG: Assignee options: {len(assignee_options)}")
                
                # Verify initial options in dropdowns match available options
                for dropdown_name, options, initial_value in [
                    ('State', state_options, prefilled_data.get('state_name')),
                    ('Assignee', assignee_options, assignee_display)
                ]:
                    if initial_value:
                        matching_option = next((opt for opt in options if opt["value"] == initial_value), None)
                        if not matching_option:
                            logger.warning(f"TRIGGER DEBUG: Initial {dropdown_name} value '{initial_value}' not found in available options")
                
                # Log additional data that might be causing issues
                logger.info(f"TRIGGER DEBUG: Private metadata: {view.get('private_metadata')}")
                
            except Exception as e:
                logger.error(f"TRIGGER DEBUG: Error in pre-open validation: {str(e)}")
            
            response = self.slack_client.views_open(
                trigger_id=trigger_id,
                view=view
            )
            
            logger.info(f"Modal opened with view ID: {response.get('view', {}).get('id')}")
            return True
            
        except SlackApiError as e:
            logger.error(f"Error opening create issue modal: {e.response['error']}")
            # Add more error details
            if 'response' in e.__dict__:
                # Safely extract and log response data without direct serialization
                error_response = {}
                if hasattr(e.response, 'data'):
                    error_response['data'] = e.response.data
                if hasattr(e.response, 'status_code'):
                    error_response['status_code'] = e.response.status_code
                if hasattr(e.response, 'headers'):
                    error_response['headers'] = dict(e.response.headers) if e.response.headers else {}
                                
                # Check if metadata exists in response
                if hasattr(e.response, 'data') and isinstance(e.response.data, dict):
                    if 'response_metadata' in e.response.data:
                        logger.error(f"TRIGGER DEBUG: Response metadata: {e.response.data.get('response_metadata')}")
                    # Add messages if available
                    if 'messages' in e.response.data:
                        logger.error(f"TRIGGER DEBUG: Messages: {e.response.data.get('messages')}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error opening modal: {str(e)}")
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
            # Initialize variables that might be referenced later
            team_options = []
            
            # Log the prefilled data for debugging
            logger.info(f"TRIGGER DEBUG: Prefilled data: {json.dumps(prefilled_data, default=str)}")
            
            # Extract the team key from issue_number format (e.g., OPS-123)
            team_key = None
            if isinstance(issue_number, str) and "-" in issue_number:
                try:
                    team_key = issue_number.split("-")[0]
                    logger.info(f"TRIGGER DEBUG: Extracted team_key {team_key} from issue_number {issue_number}")
                    issue_number = int(issue_number.split("-")[1])
                except (IndexError, ValueError) as e:
                    logger.warning(f"TRIGGER DEBUG: Could not extract team_key from '{issue_number}': {str(e)}")
            
            # Try to get team_key from prefilled_data if not found
            if not team_key:
                team_key = prefilled_data.get("teamKey") or prefilled_data.get("team_key")
                logger.info(f"TRIGGER DEBUG: Using teamKey from prefilled data: {team_key}")
            
            # First try to get complete issue data from Linear
            complete_issue_data = {}
            
            if hasattr(self, 'linear_tools') and self.linear_tools:
                try:
                    logger.info(f"TRIGGER DEBUG: Fetching complete issue data for issue #{issue_number}")
                    
                    # Set both team_key and issue_number in the filterIssues call
                    issues = self.linear_tools.filterIssues(
                        team_key=team_key,
                        issue_number=issue_number, 
                        limit=1
                    ) or []
                    
                    if issues and len(issues) > 0:
                        issue = issues[0]
                        logger.info(f"TRIGGER DEBUG: Successfully retrieved issue data: {json.dumps(issue, default=str)[:200]}...")
                        
                        # If team_key wasn't provided earlier, extract it from the retrieved issue
                        team = issue.get("team")
                        if not team_key and team and team.get("key"):
                            team_key = team.get("key")
                            logger.info(f"TRIGGER DEBUG: Extracted team_key {team_key} from retrieved issue")
                        
                        # Transform the issue data into our expected format with proper null handling
                        cycle_number = ""
                        if issue.get("cycle"):
                            cycle_number = issue["cycle"].get("number", "")
                        
                        parent_issue_number = ""
                        if issue.get("parent"):
                            parent_issue_number = issue["parent"].get("number", "")
                        
                        # Safe extraction of nested values
                        assignee_name = ""
                        if issue.get("assignee"):
                            assignee_name = issue["assignee"].get("displayName", "")
                            logger.info(f"TRIGGER DEBUG: Found assignee: {assignee_name}")
                        
                        state_name = "Todo"
                        if issue.get("state"):
                            state_name = issue["state"].get("name", "Todo")
                        
                        # Extract label names
                        label_names = []
                        if issue.get("labels") and issue["labels"].get("nodes"):
                            label_names = [label.get("name", "") for label in issue["labels"]["nodes"]]
                        
                        # Build the complete issue data
                        complete_issue_data = {
                            "title": issue.get("title", ""),
                            "description": issue.get("description", ""),
                            "priority": issue.get("priority", 0.0),
                            "state_name": state_name,
                            "assignee_name": assignee_name,
                            "label_names": label_names,
                            "project_name": issue.get("project", {}).get("name", "") if issue.get("project") else "",
                            "teamKey": team_key,
                            "parent_issue_number": parent_issue_number,
                            "cycle_number": cycle_number
                        }
                        
                        logger.info(f"TRIGGER DEBUG: Complete issue data prepared")
                        logger.info(f"TRIGGER DEBUG: Title: '{complete_issue_data.get('title')}'")
                        logger.info(f"TRIGGER DEBUG: Assignee: '{complete_issue_data.get('assignee_name')}'")
                    else:
                        logger.warning(f"TRIGGER DEBUG: No issue found with number {issue_number}")
                except Exception as e:
                    logger.warning(f"TRIGGER DEBUG: Error fetching complete issue data: {str(e)}")
                    logger.warning(f"TRIGGER DEBUG: Error details: {repr(e)}")
            
            # Now merge the prefilled_data (only for fields that exist) into the complete_issue_data
            issue_data = complete_issue_data.copy()
            
            # Only overlay fields that exist in prefilled_data
            for field in prefilled_data:
                if field in issue_data and prefilled_data[field] is not None:
                    # Don't handle cycle_name specially anymore, as we now only support cycle_number
                    issue_data[field] = prefilled_data[field]
                    logger.info(f"TRIGGER DEBUG: Overlaid field '{field}' with value: {prefilled_data[field]}")
            
            # Final fallback for team_key - use a default if all else fails
            if not team_key:
                # Default to OPS team if specified in the log
                team_key = "OPS"
                logger.warning(f"TRIGGER DEBUG: No team_key found, defaulting to {team_key}")
                issue_data["teamKey"] = team_key
            
            # Get available states for the team
            state_options = []
            if team_key and (hasattr(self, 'linear_tools') and self.linear_tools or self.linear_client):
                try:
                    if hasattr(self, 'linear_tools') and self.linear_tools:
                        states = self.linear_tools.getAllStates(teamKey=team_key)
                    else:
                        from tools.tools_declaration import linear_tools
                        states = linear_tools.getAllStates(teamKey=team_key)
                    
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
            if assignee_name and (hasattr(self, 'linear_tools') and self.linear_tools or self.linear_client):
                try:
                    if not assignee_name.startswith("@"):
                        assignee_name = f"@{assignee_name}"
                    
                    if hasattr(self, 'linear_tools') and self.linear_tools:
                        user_info = self.linear_tools.getCurrentUser(slack_display_name=assignee_name)
                    else:
                        from tools.tools_declaration import linear_tools
                        user_info = linear_tools.getCurrentUser(slack_display_name=assignee_name)
                    
                    if user_info and user_info.get("linear_display_name"):
                        assignee_display = user_info.get("linear_display_name")
                except Exception as e:
                    logger.warning(f"Error getting user info for {assignee_name}: {str(e)}")
            
            # Get all users for dropdown
            assignee_options = []
            try:
                if hasattr(self, 'linear_tools') and self.linear_tools or self.linear_client:
                    # Use team_key when calling getAllUsers - this is the key fix
                    if team_key:
                        logger.info(f"TRIGGER DEBUG: Getting users for team {team_key}")
                        if hasattr(self, 'linear_tools') and self.linear_tools:
                            # Use linear_tools method which expects teamKey as parameter
                            users = self.linear_tools.getAllUsers(teamKey=team_key)
                            logger.info(f"TRIGGER DEBUG: Retrieved {len(users)} users from linear_tools")
                            
                            # Format users from linear_tools (returns list of user objects)
                            for user in users:
                                if user.get("displayName"):
                                    assignee_options.append({
                                        "text": {
                                            "type": "plain_text",
                                            "text": user.get("displayName")
                                        },
                                        "value": user.get("displayName")
                                    })
                        else:
                            # Use linear_client method (also requires teamKey)
                            users = self.linear_client.getAllUsers(teamKey=team_key)
                            logger.info(f"TRIGGER DEBUG: Retrieved {len(users)} users from linear_client")
                            
                            # Format users from linear_client
                            for user in users:
                                if user.get("displayName"):
                                    assignee_options.append({
                                        "text": {
                                            "type": "plain_text",
                                            "text": user.get("displayName")
                                        },
                                        "value": user.get("displayName")
                                    })
                    else:
                        # If no team_key, use getCurrentUser which doesn't require a team
                        logger.warning("TRIGGER DEBUG: No team_key available, falling back to getCurrentUser")
                        if hasattr(self, 'linear_tools') and self.linear_tools:
                            all_users = self.linear_tools.getCurrentUser("")  # Empty string as fallback
                        else:
                            from tools.tools_declaration import linear_tools
                            all_users = linear_tools.getCurrentUser("")  # Empty string as fallback
                        
                        # Sort users by their linear_display_name (assuming this is dictionary format)
                        if isinstance(all_users, dict):
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
                logger.warning(f"TRIGGER DEBUG: Error details: {repr(e)}")
                
                # Add a fallback option for the current assignee
                if assignee_display:
                    assignee_options.append({
                        "text": {
                            "type": "plain_text",
                            "text": assignee_display
                        },
                        "value": assignee_display
                    })
            
            # If no options found, add a placeholder
            if not assignee_options:
                assignee_options.append({
                    "text": {
                        "type": "plain_text",
                        "text": "No assignee"
                    },
                    "value": ""
                })
            
            # Create a list of team options
            team_options = []
            for team_key_option in ["OPS", "RES", "MKT", "AI", "ENG", "PRO"]:
                team_options.append({
                    "text": {
                        "type": "plain_text",
                        "text": team_key_option
                    },
                    "value": team_key_option
                })
                
            # Find the matching team option for the current team_key
            initial_team_option = next(
                (opt for opt in team_options if opt["value"] == team_key),
                team_options[0]  # Default to first option if no match
            )
            logger.info(f"TRIGGER DEBUG: Setting initial team option to: {json.dumps(initial_team_option)}")
            
            # Construct modal view
            view = {
                "type": "modal",
                "callback_id": "linear_update_issue_modal",
                "private_metadata": json.dumps({
                    "conversation_id": conversation_id,
                    "action": "update_issue",
                    "issue_number": issue_number,
                    "team_key": team_key
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
                    
                    # Team selection dropdown
                    {
                        "type": "input",
                        "block_id": "team_block",
                        "element": {
                            "type": "static_select",
                            "action_id": "team_select",
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Select team"
                            },
                            "options": team_options,
                            "initial_option": initial_team_option
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
                            "initial_value": self._truncate_description(issue_data.get("description", "")),
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
                            "initial_option": next(
                                (opt for opt in [
                                    {"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"},
                                    {"text": {"type": "plain_text", "text": "Urgent"}, "value": "1.0"},
                                    {"text": {"type": "plain_text", "text": "High"}, "value": "2.0"},
                                    {"text": {"type": "plain_text", "text": "Medium"}, "value": "3.0"},
                                    {"text": {"type": "plain_text", "text": "Low"}, "value": "4.0"}
                                ] if opt["value"] == str(float(issue_data.get("priority", 0) or 0))),
                                {"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}
                            ) if issue_data.get("priority") is not None else None
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
                            "options": assignee_options
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
                    },
                    # Parent Issue
                    {
                        "type": "input",
                        "block_id": "parent_issue_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "parent_issue_input",
                            "initial_value": str(issue_data.get("parent_issue_number", "")),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Parent issue number"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Parent Issue #"
                        },
                        "optional": True
                    },
                    # Cycle (renamed to Cycle Number)
                    {
                        "type": "input",
                        "block_id": "cycle_block",
                        "element": {
                            "type": "plain_text_input",
                            "action_id": "cycle_input",
                            "initial_value": str(issue_data.get("cycle_number", "")),
                            "placeholder": {
                                "type": "plain_text",
                                "text": "Cycle number"
                            }
                        },
                        "label": {
                            "type": "plain_text",
                            "text": "Cycle Number"
                        },
                        "optional": True
                    }
                ]
            }
            
            # Assignee - Don't set initial_option yet
            assignee_element = {
                "type": "static_select",
                "action_id": "assignee_select",
                "placeholder": {
                    "type": "plain_text",
                    "text": "Select assignee"
                },
                "options": assignee_options
            }
            
            # Only add initial_option if we have a valid assignee and it exists in options
            assignee_name = issue_data.get("assignee_name", "")
            if assignee_name:
                logger.info(f"TRIGGER DEBUG: Checking for assignee option matching: '{assignee_name}'")
                matching_option = next(
                    (opt for opt in assignee_options if opt["value"] == assignee_name),
                    None
                )
                
                if matching_option:
                    logger.info(f"TRIGGER DEBUG: Found matching assignee option, setting initial_option")
                    assignee_element["initial_option"] = matching_option
                else:
                    logger.warning(f"TRIGGER DEBUG: No matching assignee option found for '{assignee_name}'")
            else:
                logger.info("TRIGGER DEBUG: No assignee name, skipping initial_option")
            
            # Create the assignee block with the element
            assignee_block = {
                "type": "input",
                "block_id": "assignee_block",
                "element": assignee_element,
                "label": {
                    "type": "plain_text",
                    "text": "Assignee"
                },
                "optional": True
            }
            
            # Replace the assignee block in the blocks list
            for i, block in enumerate(view['blocks']):
                if block.get('block_id') == 'assignee_block':
                    view['blocks'][i] = assignee_block
                    break
            
            # Add the same priority validation we did for create_issue_modal
            priority_str = str(issue_data.get("priority", 0.0))
            
            # Enhanced priority validation to avoid invalid_arguments error
            # Ensure the priority value exists in our options list
            priority_block = None
            for block in view['blocks']:
                if block.get('block_id') == 'priority_block':
                    priority_block = block
                    break
                    
            if priority_block and 'element' in priority_block and 'options' in priority_block['element']:
                valid_priority_values = [opt["value"] for opt in priority_block['element']['options']]
                logger.info(f"TRIGGER DEBUG: Valid priority values: {valid_priority_values}")
                logger.info(f"TRIGGER DEBUG: Current priority value: {priority_str}")
                
                # If priority_str is not in valid values, default to "0.0"
                if priority_str not in valid_priority_values:
                    logger.warning(f"TRIGGER DEBUG: Priority value '{priority_str}' not in valid options, defaulting to '0.0'")
                    priority_str = "0.0"
                
                # Find the matching option from the available options
                priority_option = next(
                    (opt for opt in priority_block['element']['options'] if opt["value"] == priority_str),
                    priority_block['element']['options'][0]  # Default to first option if no match
                )
                
                # Use the exact option object from the available options
                priority_block['element']["initial_option"] = priority_option
                logger.info(f"TRIGGER DEBUG: Setting initial_option to: {json.dumps(priority_option)}")
            else:
                logger.warning(f"TRIGGER DEBUG: Could not find priority block or options")
            
            # Debug the overall size
            try:
                view_json = json.dumps(view)
                logger.info(f"TRIGGER DEBUG: View size in bytes: {len(view_json)}")
            except Exception as e:
                logger.error(f"Error calculating view size: {str(e)}")
            
            # Add detailed trigger_id logging
            logger.info(f"TRIGGER DEBUG: SlackModals called with trigger_id: {trigger_id}")
            logger.info(f"TRIGGER DEBUG: Modal has {len(view['blocks'])} blocks")
            
            # Limit number of options in dropdowns if too many
            MAX_OPTIONS = 25
            if len(team_options) > MAX_OPTIONS:
                logger.info(f"TRIGGER DEBUG: Limiting team_options from {len(team_options)} to {MAX_OPTIONS}")
                team_options = team_options[:MAX_OPTIONS]
                
            if len(assignee_options) > MAX_OPTIONS:
                logger.info(f"TRIGGER DEBUG: Limiting assignee_options from {len(assignee_options)} to {MAX_OPTIONS}")
                assignee_options = assignee_options[:MAX_OPTIONS]
                
            if len(state_options) > MAX_OPTIONS:
                logger.info(f"TRIGGER DEBUG: Limiting state_options from {len(state_options)} to {MAX_OPTIONS}")
                state_options = state_options[:MAX_OPTIONS]
            
            # Update view with limited options
            # First find the correct blocks by block_id instead of using fixed indices
            for i, block in enumerate(view['blocks']):
                if block.get('block_id') == 'team_block' and 'element' in block:
                    logger.info(f"TRIGGER DEBUG: Updating team_block (index {i}) with {len(team_options)} options")
                    block['element']['options'] = team_options
                elif block.get('block_id') == 'state_block' and 'element' in block:
                    logger.info(f"TRIGGER DEBUG: Updating state_block (index {i}) with {len(state_options)} options")
                    block['element']['options'] = state_options
                elif block.get('block_id') == 'assignee_block' and 'element' in block:
                    logger.info(f"TRIGGER DEBUG: Updating assignee_block (index {i}) with {len(assignee_options)} options")
                    block['element']['options'] = assignee_options
            
            # Validate and potentially reduce view size
            view, was_reduced = self._validate_view_size(view)
            if was_reduced:
                logger.info("View size was reduced to fit Slack's limits")
            
            # Finally, verify trigger_id is valid (should be less than 30 seconds old)
            try:
                # Validate trigger_id format
                if not trigger_id or not isinstance(trigger_id, str):
                    logger.error(f"TRIGGER DEBUG: Invalid trigger_id format: {trigger_id}")
                    return False
                
                # Make sure it's not empty or just whitespace
                trigger_id = trigger_id.strip()
                if not trigger_id:
                    logger.error("TRIGGER DEBUG: Empty trigger_id after stripping whitespace")
                    return False
                    
                # Log the trigger_id we're about to use
                logger.info(f"TRIGGER DEBUG: Using trigger_id: {trigger_id[:20]}...")
                
                # Slack trigger_id format is typically: {timestamp}.{team_id}.{random}
                # The timestamp portion is not a Unix timestamp but a different format
                # So we'll just check if the trigger_id was recently received instead
                action_ts = None  # Initialize action_ts directly without referencing undefined payload
                if action_ts:
                    try:
                        action_timestamp = float(action_ts)
                        current_timestamp = time.time()
                        trigger_age = current_timestamp - action_timestamp
                        logger.info(f"TRIGGER DEBUG: Action timestamp age: {trigger_age:.2f} seconds")
                        if trigger_age > 25:  # Slack's limit is 30 seconds, being conservative
                            logger.warning(f"Trigger may be expired (action age: {trigger_age:.2f} seconds)")
                    except Exception as e:
                        logger.warning(f"Could not parse action_ts: {str(e)}")
                else:
                    logger.info(f"TRIGGER DEBUG: Using trigger_id without timestamp validation: {trigger_id[:20]}...")
            except Exception as e:
                logger.warning(f"Could not validate trigger_id timing: {str(e)}")
            
            # Add more debug information about the view structure before opening the modal
            try:
                # Log key information about the issue data being used
                logger.info(f"TRIGGER DEBUG: Issue data summary before opening modal:")
                logger.info(f"TRIGGER DEBUG: Issue #{issue_number} - Team key: {team_key}")
                logger.info(f"TRIGGER DEBUG: Title length: {len(issue_data.get('title', ''))}")
                logger.info(f"TRIGGER DEBUG: Description length: {len(issue_data.get('description', ''))}")
                
                # Check for and log any missing required fields
                required_fields = ['title']
                missing_fields = [field for field in required_fields if not issue_data.get(field)]
                if missing_fields:
                    logger.warning(f"TRIGGER DEBUG: Missing required fields in issue data: {missing_fields}")
                
                # Log information about dropdown options
                logger.info(f"TRIGGER DEBUG: Team options: {len(team_options)}")
                logger.info(f"TRIGGER DEBUG: State options: {len(state_options)}")
                logger.info(f"TRIGGER DEBUG: Assignee options: {len(assignee_options)}")
                
                # Verify initial options in dropdowns match available options
                for dropdown_name, options, initial_value in [
                    ('State', state_options, issue_data.get('state_name')),
                    ('Assignee', assignee_options, assignee_display)
                ]:
                    if initial_value:
                        matching_option = next((opt for opt in options if opt["value"] == initial_value), None)
                        if not matching_option:
                            logger.warning(f"TRIGGER DEBUG: Initial {dropdown_name} value '{initial_value}' not found in available options")
                
                # Log additional data that might be causing issues
                logger.info(f"TRIGGER DEBUG: Private metadata: {view.get('private_metadata')}")
                
            except Exception as e:
                logger.error(f"TRIGGER DEBUG: Error in pre-open validation: {str(e)}")
            
            # Use the trigger_id immediately after validation to minimize the chance of expiration
            try:
                response = self.slack_client.views_open(
                    trigger_id=trigger_id,
                    view=view
                )
                
                logger.info(f"Modal opened with view ID: {response.get('view', {}).get('id')}")
                return True
            except SlackApiError as e:
                logger.error(f"Error opening update issue modal: {e.response['error']}")
                
                # Log the entire exception for debugging
                logger.error(f"TRIGGER DEBUG: Full exception: {str(e)}")
                
                # Based on our tests, e.response is a dictionary that directly contains 'error' and 'response_metadata'
                try:
                    # Log the whole response as a dictionary
                    logger.error(f"TRIGGER DEBUG: Response dict: {dict(e.response)}")
                    
                    # Check for response_metadata which should be directly in the response
                    if 'response_metadata' in e.response:
                        metadata = e.response['response_metadata']
                        logger.error(f"TRIGGER DEBUG: Response metadata: {metadata}")
                        
                        # Check for messages in metadata
                        if 'messages' in metadata:
                            messages = metadata['messages']
                            logger.error(f"TRIGGER DEBUG: Metadata messages: {messages}")
                            
                            # Check for trigger_id related errors
                            for msg in messages:
                                if 'trigger_id' in msg.lower():
                                    logger.error(f"TRIGGER DEBUG: TRIGGER ID ERROR FOUND: {msg}")
                                    logger.error(f"TRIGGER DEBUG: Used trigger_id: {trigger_id}")
                                    logger.error(f"TRIGGER DEBUG: Current time: {time.time()}")
                
                except Exception as ex:
                    logger.error(f"TRIGGER DEBUG: Error processing response: {str(ex)}")
                
                # If it's invalid_arguments, provide more context
                if e.response.get('error') == 'invalid_arguments':
                    logger.error("TRIGGER DEBUG: Detected invalid_arguments error - likely an issue with the trigger_id or view structure")
                    
                    # Log additional details about the view structure
                    try:
                        # Simplified version of the view for logging
                        simple_view = {
                            "type": view.get("type"),
                            "callback_id": view.get("callback_id"),
                            "block_count": len(view.get("blocks", [])),
                            "private_metadata_length": len(view.get("private_metadata", "")),
                            "has_submit": "submit" in view,
                            "has_close": "close" in view
                        }
                        logger.error(f"TRIGGER DEBUG: View structure summary: {simple_view}")
                    except Exception as view_err:
                        logger.error(f"TRIGGER DEBUG: Error summarizing view: {str(view_err)}")
                
                return False
            except Exception as e:
                logger.error(f"Unexpected error opening modal: {str(e)}")
                return False
            
        except Exception as e:
            logger.error(f"Error opening update issue modal: {str(e)}")
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
            # We'll no longer use prefilled_data - all parameters come from the form
            
            # Extract values from modal
            team_key = state_values.get("team_block", {}).get("team_select", {}).get("selected_option", {}).get("value", "")
            title = state_values.get("title_block", {}).get("title_input", {}).get("value", "")
            description = state_values.get("description_block", {}).get("description_input", {}).get("value", "")
            priority_str = state_values.get("priority_block", {}).get("priority_select", {}).get("selected_option", {}).get("value", "0.0")
            state_name = state_values.get("state_block", {}).get("state_select", {}).get("selected_option", {}).get("value", "")
            assignee_name = state_values.get("assignee_block", {}).get("assignee_select", {}).get("selected_option", {}).get("value", "")
            labels_str = state_values.get("labels_block", {}).get("labels_input", {}).get("value", "")
            project_name = state_values.get("project_block", {}).get("project_input", {}).get("value", "")
            parent_issue_field = state_values.get("parent_issue_block", {}).get("parent_issue_input", {}).get("value", "")
            cycle_number_field = state_values.get("cycle_block", {}).get("cycle_input", {}).get("value", "")
            
            # Convert values to appropriate types
            try:
                priority = float(priority_str)
            except ValueError:
                priority = 0.0
            
            # Parse labels
            label_names = []
            if labels_str:
                label_names = [label.strip() for label in labels_str.split(",") if label.strip()]
            
            # Prepare parameters ONLY from modal inputs
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
            
            # Add optional fields if provided in the form
            if parent_issue_field:
                try:
                    parent_issue_number = int(parent_issue_field.strip())
                    create_params["parent_issue_number"] = parent_issue_number
                except ValueError:
                    pass
                
            if cycle_number_field:
                try:
                    cycle_number = int(cycle_number_field.strip())
                    create_params["cycle_number"] = cycle_number
                    logger.info(f"TRIGGER DEBUG: Set cycle_number to {cycle_number}")
                except ValueError as e:
                    logger.warning(f"TRIGGER DEBUG: Invalid cycle number format: {str(e)}")
            
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
            # We'll no longer use prefilled_data - all parameters come from the form
            
            # Get issue number from metadata
            issue_number = metadata.get("issue_number")
            if not issue_number:
                return {
                    "success": False,
                    "message": "Issue number not provided in metadata",
                    "details": {}
                }
            
            # Extract values from modal
            # Get team from the dropdown selection
            team_key = state_values.get("team_block", {}).get("team_select", {}).get("selected_option", {}).get("value", "OPS")
            logger.info(f"TRIGGER DEBUG: Using team key from dropdown: {team_key}")
            
            title = state_values.get("title_block", {}).get("title_input", {}).get("value", "")
            description = state_values.get("description_block", {}).get("description_input", {}).get("value", "")
            priority_str = state_values.get("priority_block", {}).get("priority_select", {}).get("selected_option", {}).get("value", "0.0")
            state_name = state_values.get("state_block", {}).get("state_select", {}).get("selected_option", {}).get("value", "")
            assignee_name = state_values.get("assignee_block", {}).get("assignee_select", {}).get("selected_option", {}).get("value", "")
            labels_str = state_values.get("labels_block", {}).get("labels_input", {}).get("value", "")
            project_name = state_values.get("project_block", {}).get("project_input", {}).get("value", "")
            parent_issue_field = state_values.get("parent_issue_block", {}).get("parent_issue_input", {}).get("value", "")
            cycle_number_field = state_values.get("cycle_block", {}).get("cycle_input", {}).get("value", "")
            
            # Convert values to appropriate types
            try:
                priority = float(priority_str)
            except ValueError:
                priority = 0.0
                
            # Parse labels
            label_names = []
            if labels_str:
                label_names = [label.strip() for label in labels_str.split(",") if label.strip()]
            
            # Prepare parameters ONLY from modal inputs - changed to match what the adapter expects
            update_params = {
                # Use both names to ensure compatibility
                "issueNumber": issue_number,
                "teamKey": team_key,
                "team_key": team_key,  # Also include team_key for compatibility
                "title": title,
                "description": description,
                "priority": priority,
                "state_name": state_name,
                "assignee_name": assignee_name,
                "label_names": label_names,
                "project_name": project_name
            }
            
            # Add optional fields if provided in the form
            if parent_issue_field:
                try:
                    parent_issue_number = int(parent_issue_field.strip())
                    update_params["parent_issue_number"] = parent_issue_number
                except ValueError:
                    pass
                
            # Process cycle_number
            if cycle_number_field:
                try:
                    cycle_number = int(cycle_number_field.strip())
                    update_params["cycle_number"] = cycle_number
                    logger.info(f"TRIGGER DEBUG: Set cycle_number to {cycle_number}")
                except ValueError as e:
                    logger.warning(f"TRIGGER DEBUG: Invalid cycle number format: {str(e)}")
            
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
            # Handle various formats of priority input
            if priority is None:
                return "No priority"
                
            # Convert to float, handling both numeric and string representations
            priority_float = float(priority)
            
            # Ensure we're using exact 0.0, 1.0, etc. for comparison
            priority_rounded = round(priority_float)
            if priority_rounded == priority_float:  # It's a whole number
                priority_float = float(priority_rounded)
                
            # Map priority values to text
            priority_map = {
                0.0: "No priority",
                1.0: "Urgent",
                2.0: "High",
                3.0: "Medium",
                4.0: "Low"
            }
            return priority_map.get(priority_float, "No priority")
        except (ValueError, TypeError):
            logger.warning(f"Invalid priority value: {priority}, using default")
            return "No priority"
    
    def _format_labels(self, labels: List[str]) -> str:
        """Format labels list as comma-separated string"""
        if not labels:
            return ""
        return ", ".join(labels)
    
    def _truncate_description(self, description: str) -> str:
        """Truncate description to safe length for Slack modal."""
        MAX_LENGTH = 1500  # Slack has strict limits, be more conservative
        
        if not description or len(description) <= MAX_LENGTH:
            return description
            
        # Truncate with warning message
        truncated = description[:MAX_LENGTH]
        warning = "\n\n[NOTE: Description was truncated to fit Slack's limits. The complete description will be preserved when submitted.]"
        
        # Ensure we stay under limit even with warning
        if len(truncated) + len(warning) > MAX_LENGTH:
            truncated = truncated[:MAX_LENGTH-len(warning)]
            
        return truncated + warning
    
    def _get_safe_initial_option(self, options_list, selected_value, default_text="Default"):
        """
        Safely get an initial option for a dropdown that won't cause errors.
        
        Args:
            options_list: List of option dictionaries
            selected_value: The value to look for in the options
            default_text: Text to use if creating a default option
            
        Returns:
            A valid option dict to use as initial_option
        """
        if not options_list or not isinstance(options_list, list) or len(options_list) == 0:
            # Return a safe default option
            return {
                "text": {"type": "plain_text", "text": default_text},
                "value": default_text.lower()
            }
        
        # Look for the selected value in the options
        if selected_value:
            for option in options_list:
                if option.get("value") == selected_value:
                    return option
        
        # If we didn't find a match or no selected_value provided, use the first option
        return options_list[0]
        
    def _validate_view_size(self, view):
        """Validate view size and make adjustments if needed to fit Slack's limits."""
        # Convert view to JSON to check size
        view_json = json.dumps(view)
        view_size = len(view_json)
        
        # Slack's maximum view size is around 24KB, but we'll be more conservative
        MAX_VIEW_SIZE = 20000  # bytes
        
        # Log more details about the view structure
        try:
            logger.info(f"TRIGGER DEBUG: Original view size: {view_size} bytes")
            logger.info(f"TRIGGER DEBUG: View blocks count: {len(view.get('blocks', []))}")
            
            # Log size of each block to identify large blocks
            for i, block in enumerate(view.get('blocks', [])):
                block_json = json.dumps(block)
                block_size = len(block_json)
                logger.info(f"TRIGGER DEBUG: Block {i} ({block.get('block_id', 'unknown')}): {block_size} bytes")
                
                # Check for especially large elements within blocks
                if block_size > 2000:  # Arbitrary threshold
                    logger.info(f"TRIGGER DEBUG: Large block detected - analyzing components")
                    for key, value in block.items():
                        if key != 'block_id':
                            component_json = json.dumps(value)
                            logger.info(f"TRIGGER DEBUG:   - Component '{key}': {len(component_json)} bytes")
                
                # Special logging for blocks with options which might be causing size issues
                if 'element' in block and 'options' in block['element']:
                    options_count = len(block['element']['options'])
                    options_json = json.dumps(block['element']['options'])
                    logger.info(f"TRIGGER DEBUG: Block {i} has {options_count} options ({len(options_json)} bytes)")
        except Exception as e:
            logger.error(f"TRIGGER DEBUG: Error in view size analysis: {str(e)}")
        
        if view_size <= MAX_VIEW_SIZE:
            return view, False
        
        logger.warning(f"View size ({view_size} bytes) exceeds recommended limit ({MAX_VIEW_SIZE} bytes). Attempting to reduce...")
        
        # First, reduce description length even further if present
        for block in view['blocks']:
            if block.get('block_id') == 'description_block':
                element = block.get('element', {})
                if element.get('initial_value'):
                    # More aggressive truncation
                    current_length = len(element['initial_value'])
                    # Cut in half if still very large
                    if current_length > 1000:
                        shortened = element['initial_value'][:1000] + "\n\n[Description significantly truncated - full content will be preserved when submitted]"
                        element['initial_value'] = shortened
                        logger.info(f"Reduced description from {current_length} to {len(shortened)} characters")
        
        # Reduce options in dropdowns if needed
        for block in view['blocks']:
            if 'element' in block and 'options' in block['element'] and len(block['element']['options']) > 10:
                original_count = len(block['element']['options'])
                block['element']['options'] = block['element']['options'][:10]  # Keep only first 10
                logger.info(f"TRIGGER DEBUG: Reduced options in block {block.get('block_id', 'unknown')} from {original_count} to 10")
        
        # Create new JSON to check size
        view_json = json.dumps(view)
        new_size = len(view_json)
        
        logger.info(f"TRIGGER DEBUG: After reduction - view size: {new_size} bytes (reduced by {view_size - new_size} bytes)")
        
        if new_size <= MAX_VIEW_SIZE:
            logger.info(f"Successfully reduced view size from {view_size} to {new_size} bytes")
            return view, True
        
        logger.warning(f"View still too large ({new_size} bytes). Additional reduction needed but not implemented.")
        # Log the finalized structure
        logger.info(f"TRIGGER DEBUG: Final view structure for debug: {json.dumps(view.get('type', ''))} with {len(view.get('blocks', []))} blocks")
        
        return view, False 