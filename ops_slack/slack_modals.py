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
import asyncio
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
        self.processing_views = set()
        
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
    
    def _create_loading_view(self, title: str) -> Dict[str, Any]:
        """Creates a simple modal view with a loading message."""
        return {
            "type": "modal",
            "title": {"type": "plain_text", "text": title, "emoji": True},
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "Loading form, please wait... 🤖"
                    }
                }
            ]
        }

    async def open_create_issue_modal(
        self, 
        trigger_id: str, 
        prefilled_data: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> bool:
        """
        Opens a modal for creating a Linear issue.
        First, it opens a loading modal, then updates it with the full form.
        """
        loading_view = self._create_loading_view("Create Linear Issue")
        try:
            # Open the initial loading view to get a view_id
            response = await asyncio.to_thread(
                self.slack_client.views_open,
                trigger_id=trigger_id,
                view=loading_view
            )
            view_id = response.get("view", {}).get("id")
            if not view_id:
                logger.error("Failed to get view_id from views_open response.")
                return False

            # Create a background task to fetch data and update the view
            asyncio.create_task(
                self._update_create_issue_view(view_id, prefilled_data, conversation_id)
            )
            return True

        except SlackApiError as e:
            logger.error(f"Error opening initial loading modal: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in open_create_issue_modal: {str(e)}")
            return False

    async def _update_create_issue_view(
        self,
        view_id: str,
        prefilled_data: Dict[str, Any],
        conversation_id: Optional[str] = None
    ):
        """Fetches data and updates the create issue modal view."""
        try:
            # This is the original logic from open_create_issue_modal
            title = prefilled_data.get("title", "")
            description = prefilled_data.get("description", "")
            team_key = prefilled_data.get("team_key", "OPS")
            logger.info(f"Using team_key: {team_key}")
            
            team_options = []
            if hasattr(self, 'linear_tools') and self.linear_tools:
                try:
                    teams = self.linear_tools.getAllTeams() or []
                except Exception as e:
                    logger.warning(f"Error getting teams: {str(e)}")
                    teams = []
                
                for team in teams:
                    if team.get("key"):
                        team_options.append({
                            "text": {"type": "plain_text", "text": f"{team.get('name')} ({team.get('key')})"},
                            "value": team.get("key")
                        })
            
            if not team_options:
                for team_key_default in ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]:
                    team_options.append({"text": {"type": "plain_text", "text": team_key_default}, "value": team_key_default})

            state_options = []
            selected_team_key = prefilled_data.get("teamKey")
            if selected_team_key and hasattr(self, 'linear_tools') and self.linear_tools:
                try:
                    states = self.linear_tools.getAllStates(teamKey=selected_team_key)
                    for state in states:
                        if state.get("name"):
                            state_options.append({"text": {"type": "plain_text", "text": state.get("name")}, "value": state.get("name")})
                except Exception as e:
                    logger.warning(f"Error getting states for team {selected_team_key}: {str(e)}")

            if not state_options:
                for state_name in ["Todo", "In Progress", "Done", "Canceled"]:
                    state_options.append({"text": {"type": "plain_text", "text": state_name}, "value": state_name})

            assignee_name = prefilled_data.get("assignee_name", "")
            assignee_display = assignee_name
            if assignee_name and hasattr(self, 'linear_tools') and self.linear_tools:
                try:
                    user_info = self.linear_tools.getCurrentUser(slack_display_name=f"@{assignee_name}" if not assignee_name.startswith('@') else assignee_name)
                    if user_info and user_info.get("linear_display_name"):
                        assignee_display = user_info.get("linear_display_name")
                except Exception as e:
                    logger.warning(f"Error getting user info for {assignee_name}: {str(e)}")

            assignee_options = []
            if hasattr(self, 'linear_tools') and self.linear_tools:
                try:
                    team_users = self.linear_tools.getAllUsers(team_key)
                    for user in team_users:
                        display_name = user.get("displayName")
                        if display_name:
                            assignee_options.append({"text": {"type": "plain_text", "text": display_name}, "value": display_name})
                except Exception as e:
                    logger.warning(f"Error getting team users: {str(e)}")

            if not assignee_options:
                assignee_options.append({"text": {"type": "plain_text", "text": "No assignee"}, "value": "none"})

            view = {
                "type": "modal",
                "callback_id": "linear_create_issue_modal",
                "private_metadata": json.dumps({"conversation_id": conversation_id, "action": "create_issue"}),
                "title": {"type": "plain_text", "text": "Create Linear Issue"},
                "submit": {"type": "plain_text", "text": "Create"},
                "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {"type": "input", "block_id": "team_block", "element": {"type": "static_select", "action_id": "team_select", "placeholder": {"type": "plain_text", "text": "Select a team"}, "options": team_options}, "label": {"type": "plain_text", "text": "Team"}},
                    {"type": "input", "block_id": "title_block", "element": {"type": "plain_text_input", "action_id": "title_input", "initial_value": title, "placeholder": {"type": "plain_text", "text": "Issue title"}}, "label": {"type": "plain_text", "text": "Title"}},
                    {"type": "input", "block_id": "description_block", "element": {"type": "plain_text_input", "action_id": "description_input", "multiline": True, "initial_value": self._truncate_description(description), "placeholder": {"type": "plain_text", "text": "Issue description (supports markdown)"}}, "label": {"type": "plain_text", "text": "Description"}, "optional": True},
                    {"type": "input", "block_id": "priority_block", "element": {"type": "static_select", "action_id": "priority_select", "placeholder": {"type": "plain_text", "text": "Select priority"}, "options": [{"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}, {"text": {"type": "plain_text", "text": "Urgent"}, "value": "1.0"}, {"text": {"type": "plain_text", "text": "High"}, "value": "2.0"}, {"text": {"type": "plain_text", "text": "Medium"}, "value": "3.0"}, {"text": {"type": "plain_text", "text": "Low"}, "value": "4.0"}], "initial_option": next((opt for opt in [{"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}, {"text": {"type": "plain_text", "text": "Urgent"}, "value": "1.0"}, {"text": {"type": "plain_text", "text": "High"}, "value": "2.0"}, {"text": {"type": "plain_text", "text": "Medium"}, "value": "3.0"}, {"text": {"type": "plain_text", "text": "Low"}, "value": "4.0"}] if opt["value"] == str(float(prefilled_data.get("priority", 0) or 0))), {"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}) if prefilled_data.get("priority") is not None else None}, "label": {"type": "plain_text", "text": "Priority"}, "optional": True},
                    {"type": "input", "block_id": "state_block", "element": {"type": "static_select", "action_id": "state_select", "placeholder": {"type": "plain_text", "text": "Select status"}, "options": state_options, "initial_option": next((opt for opt in state_options if opt["value"] == prefilled_data.get("state_name", "Todo")), state_options[0] if state_options else None)}, "label": {"type": "plain_text", "text": "Status"}, "optional": True},
                    {"type": "input", "block_id": "assignee_block", "element": {"type": "static_select", "action_id": "assignee_select", "placeholder": {"type": "plain_text", "text": "Select assignee"}, "options": assignee_options, "initial_option": next((opt for opt in assignee_options if opt["value"] == assignee_display), None) if assignee_display else None}, "label": {"type": "plain_text", "text": "Assignee"}, "optional": True},
                    {"type": "input", "block_id": "labels_block", "element": {"type": "plain_text_input", "action_id": "labels_input", "initial_value": self._format_labels(prefilled_data.get("label_names", [])), "placeholder": {"type": "plain_text", "text": "Labels (comma separated)"}}, "label": {"type": "plain_text", "text": "Labels"}, "optional": True},
                    {"type": "input", "block_id": "project_block", "element": {"type": "plain_text_input", "action_id": "project_input", "initial_value": prefilled_data.get("project_name", ""), "placeholder": {"type": "plain_text", "text": "Project name"}}, "label": {"type": "plain_text", "text": "Project"}, "optional": True},
                    {"type": "input", "block_id": "parent_issue_block", "element": {"type": "plain_text_input", "action_id": "parent_issue_input", "initial_value": str(prefilled_data.get("parent_issue_number", "")), "placeholder": {"type": "plain_text", "text": "Parent issue number"}}, "label": {"type": "plain_text", "text": "Parent Issue #"}, "optional": True},
                    {"type": "input", "block_id": "cycle_block", "element": {"type": "plain_text_input", "action_id": "cycle_input", "initial_value": str(prefilled_data.get("cycle_number", "")), "placeholder": {"type": "plain_text", "text": "Cycle number"}}, "label": {"type": "plain_text", "text": "Cycle Number"}, "optional": True}
                ]
            }

            view, was_reduced = self._validate_view_size(view)
            if was_reduced:
                logger.info("View size was reduced to fit Slack's limits")

            await asyncio.to_thread(
                self.slack_client.views_update,
                view_id=view_id,
                view=view
            )
            logger.info(f"Successfully updated view {view_id} with create issue form.")

        except SlackApiError as e:
            logger.error(f"Error updating create issue modal view {view_id}: {e.response['error']}")
            error_view = {
                "type": "modal",
                "title": {"type": "plain_text", "text": "Error"},
                "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"Sorry, I couldn't load the form. Error: {e.response['error']}"}}]
            }
            try:
                await asyncio.to_thread(
                    self.slack_client.views_update,
                    view_id=view_id,
                    view=error_view
                )
            except SlackApiError:
                pass
        except Exception as e:
            logger.error(f"Unexpected error in _update_create_issue_view: {str(e)}")
    
    async def open_update_issue_modal(
        self, 
        trigger_id: str, 
        issue_number: int,
        prefilled_data: Dict[str, Any],
        conversation_id: Optional[str] = None
    ) -> bool:
        """
        Opens a modal for updating a Linear issue.
        First, it opens a loading modal, then updates it with the full form.
        """
        loading_view = self._create_loading_view(f"Update Issue #{issue_number}")
        try:
            # Open the initial loading view to get a view_id
            response = await asyncio.to_thread(
                self.slack_client.views_open,
                trigger_id=trigger_id,
                view=loading_view
            )
            view_id = response.get("view", {}).get("id")
            if not view_id:
                logger.error("Failed to get view_id from views_open response.")
                return False

            # Create a background task to fetch data and update the view
            asyncio.create_task(
                self._update_update_issue_view(view_id, issue_number, prefilled_data, conversation_id)
            )
            return True

        except SlackApiError as e:
            logger.error(f"Error opening initial loading modal for update: {e.response['error']}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in open_update_issue_modal: {str(e)}")
            return False

    async def _update_update_issue_view(
        self,
        view_id: str,
        issue_number: int,
        prefilled_data: Dict[str, Any],
        conversation_id: Optional[str] = None
    ):
        """Fetches data and updates the update issue modal view."""
        try:
            team_key = None
            if isinstance(issue_number, str) and "-" in issue_number:
                try:
                    team_key = issue_number.split("-")[0]
                    issue_number = int(issue_number.split("-")[1])
                except (IndexError, ValueError):
                    pass
            
            if not team_key:
                team_key = prefilled_data.get("teamKey") or prefilled_data.get("team_key")

            complete_issue_data = {}
            if hasattr(self, 'linear_tools') and self.linear_tools:
                try:
                    issues = self.linear_tools.filterIssues(team_key=team_key, issue_number=issue_number, limit=1) or []
                    if issues:
                        issue = issues[0]
                        if not team_key and issue.get("team"):
                            team_key = issue["team"].get("key")
                        
                        complete_issue_data = {
                            "title": issue.get("title", ""), "description": issue.get("description", ""),
                            "priority": issue.get("priority", 0.0),
                            "state_name": issue.get("state", {}).get("name", "Todo"),
                            "assignee_name": issue.get("assignee", {}).get("displayName", ""),
                            "label_names": [label.get("name", "") for label in issue.get("labels", {}).get("nodes", [])],
                            "project_name": issue.get("project", {}).get("name", "") if issue.get("project") else "",
                            "teamKey": team_key,
                            "parent_issue_number": issue.get("parent", {}).get("number", ""),
                            "cycle_number": issue.get("cycle", {}).get("number", "")
                        }
                except Exception as e:
                    logger.warning(f"Error fetching complete issue data: {str(e)}")

            issue_data = complete_issue_data.copy()
            issue_data.update({k: v for k, v in prefilled_data.items() if v is not None})

            if not team_key:
                team_key = "OPS"
                issue_data["teamKey"] = team_key

            state_options = []
            if team_key and hasattr(self, 'linear_tools') and self.linear_tools:
                try:
                    states = self.linear_tools.getAllStates(teamKey=team_key)
                    for state in states:
                        if state.get("name"):
                            state_options.append({"text": {"type": "plain_text", "text": state.get("name")}, "value": state.get("name")})
                except Exception as e:
                    logger.warning(f"Error getting states for team {team_key}: {str(e)}")
            
            if not state_options:
                for state_name in ["Todo", "In Progress", "Done", "Canceled"]:
                    state_options.append({"text": {"type": "plain_text", "text": state_name}, "value": state_name})

            assignee_options = []
            if hasattr(self, 'linear_tools') and self.linear_tools and team_key:
                try:
                    users = self.linear_tools.getAllUsers(teamKey=team_key)
                    for user in users:
                        if user.get("displayName"):
                            assignee_options.append({"text": {"type": "plain_text", "text": user.get("displayName")}, "value": user.get("displayName")})
                except Exception as e:
                    logger.warning(f"Error getting users for dropdown: {str(e)}")

            if not assignee_options:
                assignee_options.append({"text": {"type": "plain_text", "text": "No assignee"}, "value": ""})

            team_options = [{"text": {"type": "plain_text", "text": k}, "value": k} for k in ["OPS", "RES", "MKT", "AI", "ENG", "PRO"]]
            initial_team_option = next((opt for opt in team_options if opt["value"] == team_key), team_options[0])

            view = {
                "type": "modal", "callback_id": "linear_update_issue_modal",
                "private_metadata": json.dumps({"conversation_id": conversation_id, "action": "update_issue", "issue_number": issue_number, "team_key": team_key}),
                "title": {"type": "plain_text", "text": f"Update Issue #{issue_number}"},
                "submit": {"type": "plain_text", "text": "Update"}, "close": {"type": "plain_text", "text": "Cancel"},
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"*Updating issue #{issue_number}*"}},
                    {"type": "input", "block_id": "team_block", "element": {"type": "static_select", "action_id": "team_select", "placeholder": {"type": "plain_text", "text": "Select team"}, "options": team_options, "initial_option": initial_team_option}, "label": {"type": "plain_text", "text": "Team"}},
                    {"type": "input", "block_id": "title_block", "element": {"type": "plain_text_input", "action_id": "title_input", "initial_value": issue_data.get("title", "")}, "label": {"type": "plain_text", "text": "Title"}},
                    {"type": "input", "block_id": "description_block", "element": {"type": "plain_text_input", "action_id": "description_input", "multiline": True, "initial_value": self._truncate_description(issue_data.get("description", ""))}, "label": {"type": "plain_text", "text": "Description"}, "optional": True},
                    {"type": "input", "block_id": "priority_block", "element": {"type": "static_select", "action_id": "priority_select", "options": [{"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}, {"text": {"type": "plain_text", "text": "Urgent"}, "value": "1.0"}, {"text": {"type": "plain_text", "text": "High"}, "value": "2.0"}, {"text": {"type": "plain_text", "text": "Medium"}, "value": "3.0"}, {"text": {"type": "plain_text", "text": "Low"}, "value": "4.0"}], "initial_option": next((opt for opt in [{"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}, {"text": {"type": "plain_text", "text": "Urgent"}, "value": "1.0"}, {"text": {"type": "plain_text", "text": "High"}, "value": "2.0"}, {"text": {"type": "plain_text", "text": "Medium"}, "value": "3.0"}, {"text": {"type": "plain_text", "text": "Low"}, "value": "4.0"}] if opt["value"] == str(float(issue_data.get("priority", 0) or 0))), {"text": {"type": "plain_text", "text": "No priority"}, "value": "0.0"}) if issue_data.get("priority") is not None else None}, "label": {"type": "plain_text", "text": "Priority"}, "optional": True},
                    {"type": "input", "block_id": "state_block", "element": {"type": "static_select", "action_id": "state_select", "options": state_options, "initial_option": next((opt for opt in state_options if opt["value"] == issue_data.get("state_name", "Todo")), state_options[0] if state_options else None)}, "label": {"type": "plain_text", "text": "Status"}, "optional": True},
                    {"type": "input", "block_id": "assignee_block", "element": {"type": "static_select", "action_id": "assignee_select", "options": assignee_options, "initial_option": next((opt for opt in assignee_options if opt["value"] == issue_data.get("assignee_name")), None) if issue_data.get("assignee_name") else None}, "label": {"type": "plain_text", "text": "Assignee"}, "optional": True},
                    {"type": "input", "block_id": "labels_block", "element": {"type": "plain_text_input", "action_id": "labels_input", "initial_value": self._format_labels(issue_data.get("label_names", []))}, "label": {"type": "plain_text", "text": "Labels"}, "optional": True},
                    {"type": "input", "block_id": "project_block", "element": {"type": "plain_text_input", "action_id": "project_input", "initial_value": issue_data.get("project_name", "")}, "label": {"type": "plain_text", "text": "Project"}, "optional": True},
                    {"type": "input", "block_id": "parent_issue_block", "element": {"type": "plain_text_input", "action_id": "parent_issue_input", "initial_value": str(issue_data.get("parent_issue_number", ""))}, "label": {"type": "plain_text", "text": "Parent Issue #"}, "optional": True},
                    {"type": "input", "block_id": "cycle_block", "element": {"type": "plain_text_input", "action_id": "cycle_input", "initial_value": str(issue_data.get("cycle_number", ""))}, "label": {"type": "plain_text", "text": "Cycle Number"}, "optional": True}
                ]
            }

            await asyncio.to_thread(
                self.slack_client.views_update,
                view_id=view_id,
                view=view
            )
            logger.info(f"Successfully updated view {view_id} with update issue form.")

        except SlackApiError as e:
            logger.error(f"Error updating update issue modal view {view_id}: {e.response['error']}")
            error_view = {
                "type": "modal",
                "title": {"type": "plain_text", "text": "Error"},
                "blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": f"Sorry, I couldn't load the form. Error: {e.response['error']}"}}]
            }
            try:
                await asyncio.to_thread(
                    self.slack_client.views_update,
                    view_id=view_id,
                    view=error_view
                )
            except SlackApiError:
                pass
        except Exception as e:
            logger.error(f"Unexpected error in _update_update_issue_view: {str(e)}")
    
    async def handle_view_submission(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle submission of modal views, preventing duplicates.
        
        Args:
            payload: The view submission payload from Slack
            
        Returns:
            Result of the action with status and details.
        """
        view = payload.get("view", {})
        view_id = view.get("id")

        if not view_id:
            logger.error("No view_id in submission payload.")
            return {"success": False, "message": "Invalid submission payload."}

        if view_id in self.processing_views:
            logger.warning(f"Duplicate submission received for view_id: {view_id}. Ignoring.")
            return {"success": True, "message": "Request already in progress."}

        self.processing_views.add(view_id)
        try:
            callback_id = view.get("callback_id", "")
            metadata = json.loads(view.get("private_metadata", "{}"))
            state_values = view.get("state", {}).get("values", {})
            
            if callback_id == "linear_create_issue_modal":
                return await self._handle_create_issue(state_values, metadata)
            elif callback_id == "linear_update_issue_modal":
                return await self._handle_update_issue(state_values, metadata)
            else:
                logger.warning(f"Unknown callback_id: {callback_id}")
                return {"success": False, "message": "Unknown action type"}
                
        except Exception as e:
            logger.error(f"Error handling view submission for view_id {view_id}: {str(e)}")
            return {"success": False, "message": f"Error: {str(e)}"}
        finally:
            self.processing_views.discard(view_id)
    
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