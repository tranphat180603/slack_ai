"""
Centralized tools declaration for TMAI Agent.
This module imports and exposes all available tools from different sources.
"""

import os
import logging
import inspect
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dotenv import load_dotenv
from functools import wraps
import traceback
import json
import sys

# Ensure ops_posthog is in the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ops_posthog"))
from ops_posthog.posthog_client import PosthogClient

load_dotenv()

# Configure logger with a more explicit setup
logger = logging.getLogger("tools_declaration")
# Ensure the logger level is set to INFO or lower to see informational messages
logger.setLevel(logging.INFO)

# Add a console handler if none exists
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

logger.info("tools_declaration module loaded with explicit logging configuration")

# Import Linear tools
from ops_linear_db.linear_client import (
    LinearClient, 
    LinearError,
    LinearAuthError, 
    LinearNotFoundError, 
    LinearValidationError
)

# Import Linear RAG embedding tools
from ops_linear_db.linear_rag_embeddings import (
    semantic_search,
    store_issue_embedding,
    store_project_embedding,
    store_comment_embedding,
    get_embedding
)

# Import Website tools
from ops_website_db.website_db import WebsiteDB

# Import Slack tools
from ops_slack.slack_tools import SlackClient

from ops_gdrive.gdrive_tools import GoogleDriveClient

# Parameter adapter for Linear methods
class LinearParameterAdapter:
    """Handles conversion of flat parameters to nested structures for Linear API"""
    
    @staticmethod
    def adapt_filter_issues(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat filterIssues parameters to criteria object.
        
        Args:
            params: Dictionary of parameters:
                - team_key: Team key (enum: ENG, OPS, RES, AI, MKT, PRO)
                - issue_number: Issue number to filter by
                - state: State name (e.g. 'Todo', 'In Progress', 'Done')
                - priority: Priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)
                - assignee_name: Assignee display name (exact match)
                - assignee_contains: Assignee name contains (case-insensitive)
                - title_contains: Title contains text (case-insensitive)
                - description_contains: Description contains text (case-insensitive)
                - cycle_number: Cycle number
                - project_id: Project ID
                - label_name: Label name
                - first/limit: Maximum number of issues to return
                - include_description: Whether to include description in results
                
        Returns:
            Dictionary with adapted parameters
        """
        if 'criteria' in params and params['criteria'] is not None:
            return params
            
        criteria = {}
        
        # Team key
        if params.get("team_key") or params.get("teamKey"):
            team_key = params.get("team_key") or params.get("teamKey")
            criteria["team"] = {"key": {"eq": team_key}}
        
        # Issue number filter    
        if params.get("issue_number") or params.get("issueNumber"):
            issue_number = params.get("issue_number") or params.get("issueNumber")
            criteria["number"] = {"eq": issue_number}
            
        # Add state filter if provided
        if params.get("state") and params["state"].strip():
            criteria["state"] = {"name": {"eq": params["state"]}}
        
        # Add priority filter if provided and not zero
        # Note: In Linear, priority 0 is valid but we'll skip it if it's exactly 0.0
        if params.get("priority") is not None and params["priority"] != 0 and params["priority"] != 0.0:
            criteria["priority"] = {"eq": float(params["priority"])}
        
        # Handle assignee filters - exact match vs contains
        if params.get("assignee_name") and params["assignee_name"].strip():
            criteria["assignee"] = {"displayName": {"eq": params["assignee_name"]}}
        elif params.get("assignee_contains") and params["assignee_contains"].strip():
            criteria["assignee"] = {"displayName": {"containsIgnoreCase": params["assignee_contains"]}}
            
        # Add cycle filter if provided
        if params.get("cycle_number") and params["cycle_number"] > 0:
            criteria["cycle"] = {"number": {"eq": params["cycle_number"]}}
            
        # Add project filter if provided
        if params.get("project_id") and params["project_id"].strip():
            criteria["project"] = {"id": {"eq": params["project_id"]}}
        
        # Add label filter if provided
        if params.get("label_name") and params["label_name"].strip():
            # For label, we use a different structure that finds issues with ANY label matching the name
            criteria["labels"] = {"some": {"name": {"eq": params["label_name"]}}}
        
        # Add title contains filter
        if params.get("title_contains") and params["title_contains"].strip():
            criteria["title"] = {"containsIgnoreCase": params["title_contains"]}
            
        # Add description contains filter
        if params.get("description_contains") and params["description_contains"].strip():
            criteria["description"] = {"containsIgnoreCase": params["description_contains"]}
        
        result = {"criteria": criteria}
        
        # Add limit if provided
        if params.get("limit") is not None and params["limit"] > 0:
            result["limit"] = params["limit"]
        elif params.get("first") is not None and params["first"] > 0:
            result["limit"] = params["first"]
            
        # Pass through include_description parameter
        if "include_description" in params:
            result["include_description"] = params["include_description"]
        
        return result
    
    @staticmethod
    def adapt_filter_users(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat filterUsers parameters to criteria object"""
        if 'criteria' in params and params['criteria'] is not None:
            return params
            
        criteria = {}
        
        # Map parameters
        if 'display_name' in params and params['display_name'] is not None and params['display_name'] != '':
            criteria['displayName'] = {'eq': params['display_name']}
        elif 'display_name_contains' in params and params['display_name_contains'] is not None and params['display_name_contains'] != '':
            criteria['displayName'] = {'containsIgnoreCase': params['display_name_contains']}
            
        if 'email' in params and params['email'] is not None and params['email'] != '':
            criteria['email'] = {'eq': params['email']}
        elif 'email_contains' in params and params['email_contains'] is not None and params['email_contains'] != '':
            criteria['email'] = {'containsIgnoreCase': params['email_contains']}
            
        if 'is_active' in params and params['is_active'] is not None:
            criteria['active'] = {'eq': params['is_active']}
        
        return {'criteria': criteria}
    
    @staticmethod
    def adapt_filter_projects(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat filterProjects parameters to criteria object"""
        if 'criteria' in params and params['criteria'] is not None:
            return params
            
        criteria = {}
        
        # Map parameters based on actual GraphQL fields
        # Note: 'team' and 'teams' are not valid filter fields in ProjectFilter type
        # Team filtering will be done client-side after API call
            
        if 'name' in params and params['name'] is not None and params['name'] != '':
            criteria['name'] = {'eq': params['name']}
        elif 'name_contains' in params and params['name_contains'] is not None and params['name_contains'] != '':
            criteria['name'] = {'containsIgnoreCase': params['name_contains']}
            
        if 'state' in params and params['state'] is not None and params['state'] != '':
            criteria['state'] = {'eq': params['state']}
            
        # Adjust lead_name to match the field in GraphQL query (lead.displayName)
        if 'lead_display_name' in params and params['lead_display_name'] is not None and params['lead_display_name'] != '':
            criteria['lead'] = {'displayName': {'eq': params['lead_display_name']}}
        elif 'lead_contains' in params and params['lead_contains'] is not None and params['lead_contains'] != '':
            criteria['lead'] = {'displayName': {'containsIgnoreCase': params['lead_contains']}}
        
        # Store original teamKey parameter for client-side filtering
        teamKey = params.get('teamKey')
        
        return {'criteria': criteria, 'teamKey': teamKey}
    
    @staticmethod
    def adapt_filter_cycles(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat filterCycles parameters to criteria object"""
        if 'criteria' in params and params['criteria'] is not None:
            return params
            
        criteria = {}
        
        # Map parameters based on actual GraphQL fields
        if 'teamKey' in params and params['teamKey'] is not None and params['teamKey'] != '':
            criteria['team'] = {'key': {'eq': params['teamKey']}}
        
        # Cycle GraphQL query doesn't include 'name' field, only 'number'
        if 'number' in params and params['number'] is not None and params['number'] != 0:
            try:
                cycle_number = int(params['number'])
                criteria['number'] = {'eq': cycle_number}
            except (ValueError, TypeError):
                pass
        
        # Handle name contains (will be matched against non-GraphQL field client-side)
        if 'name_contains' in params and params['name_contains'] is not None and params['name_contains'] != '':
            # Store for client-side filtering, not in criteria object
            params['client_side_name_contains'] = params['name_contains']
            
        # Handle date filters which are in the GraphQL query
        if 'starts_at' in params and params['starts_at'] is not None and params['starts_at'] != '':
            criteria['startsAt'] = {'eq': params['starts_at']}
        
        if 'ends_at' in params and params['ends_at'] is not None and params['ends_at'] != '':
            criteria['endsAt'] = {'eq': params['ends_at']}
        
        # Get filter_by_start_date but don't add it to criteria
        filter_by_start_date = params.get('filter_by_start_date')
        if filter_by_start_date is None:  # If not specified, use default True
            filter_by_start_date = True
        
        # Preserve filter_by_start_date parameter and any client-side filtering params
        result = {
            'criteria': criteria,
            'filter_by_start_date': filter_by_start_date
        }
        
        # Add client_side_name_contains if present
        if 'client_side_name_contains' in params:
            result['client_side_name_contains'] = params['client_side_name_contains']
            
        return result
    
    @staticmethod
    def adapt_filter_comments(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat filterComments parameters to criteria object"""
        if 'criteria' in params and params['criteria'] is not None:
            return params
            
        criteria = {}
        
        # Map parameters to match GraphQL query fields
        if 'issue_number' in params and params['issue_number'] is not None and params['issue_number'] != 0:
            criteria['issue'] = {'number': {'eq': params['issue_number']}}
            
        # Body contains is tested against body field in GraphQL
        if 'body_contains' in params and params['body_contains'] is not None and params['body_contains'] != '':
            criteria['body'] = {'containsIgnoreCase': params['body_contains']}
            
        # User displayName is the correct field in GraphQL
        if 'user_display_name' in params and params['user_display_name'] is not None and params['user_display_name'] != '':
            criteria['user'] = {'displayName': {'eq': params['user_display_name']}}
        elif 'user_contains' in params and params['user_contains'] is not None and params['user_contains'] != '':
            criteria['user'] = {'displayName': {'containsIgnoreCase': params['user_contains']}}
        
        return {'criteria': criteria}
    
    @staticmethod
    def adapt_filter_attachments(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat filterAttachments parameters to criteria object for client-side filtering"""
        if 'criteria' in params and params['criteria'] is not None:
            return params
            
        criteria = {}
        
        # Simply pass the parameters through in a flat structure for client-side filtering
        if 'issue_number' in params and params['issue_number'] is not None and params['issue_number'] != 0:
            criteria['issue_number'] = params['issue_number']
            
        # Title contains will be filtered client-side
        if 'title_contains' in params and params['title_contains'] is not None and params['title_contains'] != '':
            criteria['title'] = {'containsIgnoreCase': params['title_contains']}
            
        # Creator displayName will be filtered client-side
        if 'creator_display_name' in params and params['creator_display_name'] is not None and params['creator_display_name'] != '':
            criteria['creator'] = {'displayName': {'eq': params['creator_display_name']}}
        elif 'creator_contains' in params and params['creator_contains'] is not None and params['creator_contains'] != '':
            criteria['creator'] = {'displayName': {'containsIgnoreCase': params['creator_contains']}}
        
        return {'criteria': criteria}
    
    @staticmethod
    def adapt_create_issue(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat createIssue parameters to data object.
        Automatically looks up IDs for teamKey, assignee_name, etc.
        """
        import logging  # Import logging at the function level to fix the error
        
        if 'data' in params and params['data'] is not None:
            return params
            
        # Extract the issue data from flat parameters
        data = {}
        linear_client = None
        
        # Map direct field copies
        for field in ['teamId', 'title', 'description', 'priority', 'estimate',
                     'stateId', 'assigneeId', 'labelIds', 'cycleId', 'projectId', 'parentId']:
            if field in params and params[field] is not None:
                data[field] = params[field]
        
        # Initialize LinearClient if needed for lookups
        if linear_client is None and ('teamKey' in params or 'assignee_name' in params 
                                    or 'state_name' in params or 'label_names' in params):
            try:
                from ops_linear_db.linear_client import LinearClient
                import os
                linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error initializing LinearClient: {str(e)}")
                linear_client = None
        
        # Handle team_key to teamId conversion
        if 'teamKey' in params and params['teamKey'] and 'teamId' not in data:
            try:
                team_key = params['teamKey']
                
                # First approach: If we know the user, use their team
                if 'assignee_name' in params and params['assignee_name'] and linear_client:
                    # Try to get user info from getCurrentUser
                    user_info = linear_client.getCurrentUser(f"@{params['assignee_name']}")
                    if user_info and user_info.get('team') == team_key:
                        # Get any team entity to extract its ID
                        states = linear_client.getAllStates(team_key)
                        if states and len(states) > 0:
                            team_id = states[0].get('team', {}).get('id')
                            if team_id:
                                data['teamId'] = team_id
                
                # Second approach: Get team ID from any issues in the team
                if 'teamId' not in data and linear_client:
                    issue_criteria = {"team": {"key": {"eq": team_key}}}
                    issues = linear_client.filterIssues(issue_criteria, limit=1)
                    
                    if issues and len(issues) > 0:
                        team = issues[0].get('team', {})
                        if team and 'id' in team:
                            data['teamId'] = team['id']
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up team ID: {str(e)}")
        
        # Handle assignee_name to assigneeId conversion
        if 'assignee_name' in params and params['assignee_name'] and 'assigneeId' not in data:
            try:
                # Initialize LinearClient if needed
                if linear_client is None:
                    from ops_linear_db.linear_client import LinearClient
                    import os
                    linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
                
                # First try: Use the new getUserByName function with team context
                # For update operations, we've likely already looked up the team key
                lookup_team_key = params.get('teamKey') or params.get('teamKey')
                if lookup_team_key:
                    # Look up user in the specific team first
                    user = linear_client.getUserByName(params['assignee_name'], lookup_team_key)
                    if user and user.get('id'):
                        data['assigneeId'] = user.get('id')
                        logging.getLogger("tools_declaration").info(
                            f"Found assignee '{params['assignee_name']}' in team '{lookup_team_key}' with ID: {user.get('id')}"
                        )
                
                # Second try: Use the new function without team context if not found
                if 'assigneeId' not in data:
                    user = linear_client.getUserByName(params['assignee_name'])
                    if user and user.get('id'):
                        data['assigneeId'] = user.get('id')
                        logging.getLogger("tools_declaration").info(
                            f"Found assignee '{params['assignee_name']}' with ID: {user.get('id')}"
                        )
                    else:
                        logging.getLogger("tools_declaration").warning(
                            f"Could not find assignee with name '{params['assignee_name']}'"
                        )
                
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up assignee ID: {str(e)}")
                logging.getLogger("tools_declaration").error(traceback.format_exc())
        
        # Handle state_name to stateId conversion
        if 'state_name' in params and params['state_name'] and 'stateId' not in data:
            try:
                team_key = params.get('teamKey')
                if team_key and linear_client is None:
                    from ops_linear_db.linear_client import LinearClient
                    import os
                    linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
                
                if team_key and linear_client:
                    states = linear_client.getAllStates(team_key)
                    state = next((s for s in states 
                                if s.get('name', '').lower() == params['state_name'].lower()), None)
                    
                    if state and 'id' in state:
                        data['stateId'] = state['id']
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up state ID: {str(e)}")
        
        # Handle cycle_name/cycle_number to cycleId conversion
        if ('cycle_name' in params and params['cycle_name'] and 'cycleId' not in data) or \
           ('cycle_number' in params and params['cycle_number'] and 'cycleId' not in data):
            try:
                team_key = params.get('teamKey')
                cycle_name = params.get('cycle_name')
                cycle_number = params.get('cycle_number')
                
                # If cycle_name is numeric, try to use it as a cycle_number
                if cycle_name and cycle_name.isdigit():
                    cycle_number = int(cycle_name)
                    cycle_name = None
                    logging.getLogger("tools_declaration").info(f"Converting numeric cycle_name '{params['cycle_name']}' to cycle_number {cycle_number}")
                
                if team_key and linear_client is None:
                    from ops_linear_db.linear_client import LinearClient
                    import os
                    linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
                
                if team_key and linear_client:
                    cycles = linear_client.getAllCycles(team_key)
                    cycle = None
                    
                    if cycle_name:
                        cycle = next((c for c in cycles 
                                    if c.get('name', '').lower() == cycle_name.lower()), None)
                    elif cycle_number:
                        cycle = next((c for c in cycles 
                                    if c.get('number') == cycle_number), None)
                    
                    if cycle and 'id' in cycle:
                        data['cycleId'] = cycle['id']
                        logging.getLogger("tools_declaration").info(f"Found cycle ID: {cycle['id']} for {'cycle_name' if cycle_name else 'cycle_number'}: {cycle_name if cycle_name else cycle_number}")
                    else:
                        logging.getLogger("tools_declaration").warning(f"No cycle found for {'cycle_name' if cycle_name else 'cycle_number'}: {cycle_name if cycle_name else cycle_number}")
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up cycle ID: {str(e)}")
        
        # Handle project_name to projectId conversion
        if 'project_name' in params and params['project_name'] and 'projectId' not in data:
            try:
                team_key = params.get('teamKey')
                if team_key and linear_client is None:
                    from ops_linear_db.linear_client import LinearClient
                    import os
                    linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
                
                if team_key and linear_client:
                    projects = linear_client.getAllProjects(team_key)
                    project = next((p for p in projects 
                                  if p.get('name', '').lower() == params['project_name'].lower()), None)
                    
                    if project and 'id' in project:
                        data['projectId'] = project['id']
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up project ID: {str(e)}")
        
        # Handle label_names to labelIds conversion
        if 'label_names' in params and params['label_names'] and 'labelIds' not in data:
            try:
                team_key = params.get('teamKey')
                if team_key and linear_client is None:
                    from ops_linear_db.linear_client import LinearClient
                    import os
                    linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
                
                if team_key and linear_client:
                    labels = linear_client.getAllLabels(team_key)
                    label_ids = []
                    
                    for label_name in params['label_names']:
                        label = next((l for l in labels 
                                     if l.get('name', '').lower() == label_name.lower()), None)
                        if label and 'id' in label:
                            label_ids.append(label['id'])
                    
                    if label_ids:
                        data['labelIds'] = label_ids
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up label IDs: {str(e)}")
        
        # Handle parent_issue_number to parentId conversion
        if 'parent_issue_number' in params and params['parent_issue_number'] and 'parentId' not in data:
            try:
                if linear_client is None:
                    from ops_linear_db.linear_client import LinearClient
                    import os
                    linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
                
                # Try to find the parent issue
                issue_criteria = {"number": {"eq": params['parent_issue_number']}}
                if 'teamKey' in params and params['teamKey']:
                    issue_criteria["team"] = {"key": {"eq": params['teamKey']}}
                
                issues = linear_client.filterIssues(issue_criteria)
                
                if issues and len(issues) > 0:
                    data['parentId'] = issues[0].get('id')
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up parent issue ID: {str(e)}")
                
        return {'data': data}
    
    @staticmethod
    def adapt_update_issue(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt parameters for updateIssue
        """
        # Fallback name for compatibility
        issue_number = params.get("issue_number") or params.get("issueNumber")
        
        # Base parameters
        adapted = {
            "issueNumber": issue_number,
            "teamKey": params.get("team_key") or params.get("teamKey")
        }
        
        # Set issue data to be updated
        issue_data = {}
        
        # Add title if provided
        if params.get("title") is not None:
            issue_data["title"] = params["title"]
            
        # Add description if provided
        if params.get("description") is not None:
            issue_data["description"] = params["description"]
            
        # Add priority if provided (as float)
        if params.get("priority") is not None:
            try:
                issue_data["priority"] = float(params["priority"])
            except (TypeError, ValueError):
                pass
                
        # Add estimate if provided
        if params.get("estimate") is not None:
            issue_data["estimate"] = params["estimate"]
                
        # Add state_name if provided (must look up state ID)
        if params.get("state_name") is not None:
            issue_data["stateId"] = "$getStateIdByName"
                
        # Add assignee if provided (must look up user ID)
        if params.get("assignee_name") is not None:
            issue_data["assigneeId"] = "$getUserIdByName"
                
        # Add label names if provided (must look up label IDs)
        if params.get("label_names") is not None:
            # Handle empty list case explicitly
            if isinstance(params.get("label_names"), list) and len(params.get("label_names")) == 0:
                issue_data["labelIds"] = []
            else:
                issue_data["labelIds"] = "$getLabelIdsByNames"
                
        # Add project if provided (must look up project ID)
        if params.get("project_name") is not None:
            issue_data["projectId"] = "$getProjectIdByName"
                
        # Add cycle if provided (must look up cycle ID by number)
        # Note: cycle_name is no longer supported, only cycle_number
        if params.get("cycle_number") is not None:
            issue_data["cycleId"] = "$getCycleIdByNumber"
                
        # Add parent issue if provided (must look up issue ID)
        if params.get("parent_issue_number") is not None:
            issue_data["parentId"] = "$getIssueIdByNumber"
                
        # Add archived status if provided
        if params.get("archived") is not None:
            issue_data["archived"] = params["archived"]
                
        # Set the issue data
        adapted["data"] = issue_data
            
        return adapted
    
    @staticmethod
    def adapt_create_comment(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat createComment parameters to proper format.
        Automatically looks up issue ID from issue number and team key.
        """
        import logging  # Import logging at the function level to fix the error
        logger = logging.getLogger("linear_tools")
        
        # If issue_id is already provided, use it directly
        if 'issue_id' in params or 'issueId' in params:
            issue_id = params.get('issue_id') or params.get('issueId')
            comment_data = params.get('commentData', {})
            if 'body' in params:
                comment_data['body'] = params['body']
            return {'issue_id': issue_id, 'commentData': comment_data}
            
        # Extract issue number and team key
        issue_number = params.get('issue_number') or params.get('issueNumber')
        team_key = params.get('team_key') or params.get('teamKey')
        
        # Extract comment data
        comment_data = {}
        if 'body' in params:
            comment_data['body'] = params['body']
        elif 'commentData' in params and isinstance(params['commentData'], dict):
            comment_data = params['commentData']
        
        # If we have both issue number and team key, look up the issue ID
        if issue_number and team_key:
            try:
                from ops_linear_db.linear_client import LinearClient
                # Get a linear client instance
                if hasattr(LinearParameterAdapter, 'linear_client') and LinearParameterAdapter.linear_client:
                    linear_client = LinearParameterAdapter.linear_client
                else:
                    linear_api_key = os.getenv("LINEAR_API_KEY")
                    linear_client = LinearClient(linear_api_key)
                    LinearParameterAdapter.linear_client = linear_client
                
                # Look up the issue ID using filterIssues
                criteria = {
                    "team": {"key": {"eq": team_key}},
                    "number": {"eq": issue_number}
                }
                issues = linear_client.filterIssues(criteria, limit=1)
                
                if issues and len(issues) > 0:
                    issue_id = issues[0].get("id")
                    logger.info(f"Found issue ID {issue_id} for issue #{issue_number} in team {team_key}")
                    return {'issue_id': issue_id, 'commentData': comment_data}
                else:
                    logger.warning(f"Issue #{issue_number} not found in team {team_key}")
            except Exception as e:
                logger.error(f"Error looking up issue ID: {str(e)}")
                
        # Fallback: return parameters as provided
        return params
    
    @staticmethod
    def adapt_create_attachment(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat createAttachment parameters to proper format.
        Automatically looks up issue ID from issue number.
        """
        import logging  # Import logging at the function level to fix the error
        
        if 'attachmentData' in params and params['attachmentData'] is not None:
            return params
            
        issueNumber = params.get('issueNumber') or params.get('issue_number')
        attachmentData = {}
        linear_client = None
        
        # Extract attachment fields
        for field in ['title', 'url']:
            if field in params and params[field] is not None:
                attachmentData[field] = params[field]
        
        # Look up the issue ID from the issue number
        issue_id = None
        if issueNumber:
            try:
                from ops_linear_db.linear_client import LinearClient
                import os
                linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
                
                # Find the issue by number
                issue_criteria = {"number": {"eq": issueNumber}}
                issues = linear_client.filterIssues(issue_criteria)
                
                if issues and len(issues) > 0:
                    issue_id = issues[0].get('id')
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up issue ID: {str(e)}")
                
        return {'issueNumber': issueNumber, 'issueId': issue_id, 'attachmentData': attachmentData}


# Function decorator to adapt parameters using LinearParameterAdapter
def adapt_parameters(adapter_method: Callable):
    """Decorator that adapts parameters using the specified adapter method"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get the class instance (self)
            self = args[0]
            
            # Combine args and kwargs into a single params dict
            arg_names = list(inspect.signature(func).parameters.keys())[1:]  # Skip 'self'
            params = {}
            
            # Add positional arguments
            for i, arg in enumerate(args[1:], 0):
                if i < len(arg_names):
                    params[arg_names[i]] = arg
            
            # Add keyword arguments
            params.update(kwargs)
            
            # Adapt parameters
            try:
                adapted_params = adapter_method(params)
                
                # Log the adapted parameters before calling the function
                if adapter_method.__name__ == 'adapt_update_issue':
                    logger.info(f"After adapter: adapted_params = {adapted_params}")
                
                # IMPORTANT CHANGE: Add all original parameters to adapted_params
                # This ensures critical values from original call are available
                if adapter_method.__name__ == 'adapt_update_issue':
                    # Keep all original parameters except those that would conflict with adapted ones
                    for key, value in params.items():
                        if key not in adapted_params and key != 'data':
                            adapted_params[key] = value
                    logger.info(f"Enhanced params (with originals): {adapted_params}")
                
                # Call the function with adapted parameters
                result = func(self, **adapted_params)
                
                # Special handling for createIssue to extract issue from result
                if func.__name__ == 'createIssue' and isinstance(result, dict):
                    issue = result.get('issue')
                    if issue:
                        return issue
                
                return result
            except Exception as e:
                logger.error(f"Error adapting parameters: {str(e)}")
                # Fall back to original call if adaptation fails
                return func(self, **kwargs)
                
        return wrapper
    return decorator


# Linear Tool Functions
class LinearTools:
    """Wrapper for Linear API tools"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Linear tools with API key.
        
        Args:
            api_key: Linear API key (defaults to LINEAR_API_KEY environment variable)
        """
        if api_key is None:
            api_key = os.environ.get("LINEAR_API_KEY")
            
        if not api_key:
            logger.warning("No Linear API key provided. Linear functionality will be limited.")
            self.client = None
        else:
            try:
                self.client = LinearClient(api_key)
                logger.info("Linear client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Linear client: {str(e)}")
                self.client = None
    
    def _check_client(self):
        """Check if client is initialized"""
        if not self.client:
            raise ValueError("Linear client not initialized. Please provide a valid API key.")
    
    # Team data retrieval methods
    def getAllUsers(self, teamKey: str) -> List[dict]:
        """Get all users for a team"""
        self._check_client()
        return self.client.getAllUsers(teamKey)
    
    def getAllTeams(self) -> List[dict]:
        """Get all teams"""
        self._check_client()
        return self.client.getAllTeams()
    
    def getAllIssues(self, teamKey: str, limit: int = 3, filters: dict = None) -> List[dict]:
        """Get all issues for a team"""
        self._check_client()
        return self.client.getAllIssues(teamKey, limit, filters)
    
    def getAllProjects(self, teamKey: str) -> List[dict]:
        """Get all projects for a team"""
        self._check_client()
        return self.client.getAllProjects(teamKey)
    
    def getAllCycles(self, teamKey: str, filter_by_start_date: bool = True) -> List[dict]:
        """Get all cycles for a team"""
        self._check_client()
        return self.client.getAllCycles(teamKey, filter_by_start_date)
    
    def getAllLabels(self, teamKey: str) -> List[dict]:
        """Get all labels for a team"""
        self._check_client()
        return self.client.getAllLabels(teamKey)
    
    def getAllStates(self, teamKey: str) -> List[dict]:
        """Get all workflow states for a team"""
        self._check_client()
        return self.client.getAllStates(teamKey)
    
    def getCurrentUser(self, slack_display_name: str) -> dict:
        if "@" not in slack_display_name:
            slack_display_name = "@" + slack_display_name
        """Get information about a specific user by their Slack display name"""
        self._check_client()
        return self.client.getCurrentUser(slack_display_name)
    
    def getUserMessageByNumber(self, number: int) -> list:
        """
        Retrieve the most recent N user messages from conversation history.
        
        Args:
            number: The number of most recent user messages to retrieve
            
        Returns:
            List of the most recent user messages content or an empty list if none found
        """
        self._check_client()
        
        # The conversation history needs to be provided by the calling code
        # This typically happens in the execution framework (in soldier.py or similar)
        # We'll use a placeholder value that should be replaced by the actual conversation history
        # in the execution framework
        conversation_history = []
        
        # Access the context defined by the Slack agent
        try:
            # Try different import paths for context_manager
            try:
                from app.context_manager import context_manager
            except ImportError:
                try:
                    from context_manager import context_manager
                except ImportError:
                    import sys
                    import os
                    # Try to find the root of the project to correctly import
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    parent_dir = os.path.dirname(current_dir)
                    sys.path.append(parent_dir)
                    
                    try:
                        from app.context_manager import context_manager
                    except ImportError:
                        logger.warning("Could not import context_manager from any path")
                        context_manager = None
            
            if context_manager and hasattr(context_manager, 'current_context_id'):
                current_context = context_manager.get_context(context_manager.current_context_id)
                if current_context and 'conversation_history' in current_context:
                    conversation_history = current_context['conversation_history']
        except Exception as e:
            logger.warning(f"Error accessing conversation history: {str(e)}")
            context_manager = None
            
        return self.client.getUserMessageByNumber(number, conversation_history)
    
    # Filter methods
    @adapt_parameters(LinearParameterAdapter.adapt_filter_issues)
    def filterIssues(self, criteria: dict = None, limit: int = None, include_description: bool = True, **kwargs) -> List[dict]:
        """
        Filter issues by criteria or individual parameters
        
        Args:
            criteria: Dictionary of filter criteria
            limit: Maximum number of issues to return
            include_description: If False, strip description field from results
            **kwargs: Additional filter parameters
        """
        self._check_client()
        return self.client.filterIssues(criteria, limit, include_description)
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_users)
    def filterUsers(self, criteria: dict = None, **kwargs) -> List[dict]:
        """Filter users by criteria or individual parameters"""
        self._check_client()
        return self.client.filterUsers(criteria)
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_projects)
    def filterProjects(self, criteria: dict = None, teamKey: str = None, **kwargs) -> List[dict]:
        """
        Filter projects by criteria or individual parameters
        
        Args:
            criteria: Dictionary of filter criteria
            teamKey: Team key to filter by (client-side filtering)
            **kwargs: Additional filter parameters
        
        Returns:
            List of matching project objects
        """
        self._check_client()
        
        # Get projects based on supported API filters
        projects = self.client.filterProjects(criteria)
        
        # If teamKey is provided, filter by team client-side
        if teamKey:
            filtered_projects = []
            for project in projects:
                teams = project.get("teams", {}).get("nodes", [])
                if any(team.get("key") == teamKey for team in teams):
                    filtered_projects.append(project)
            return filtered_projects
            
        return projects
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_cycles)
    def filterCycles(self, criteria: dict = None, filter_by_start_date: bool = True, client_side_name_contains: str = None, **kwargs) -> List[dict]:
        """
        Filter cycles by criteria or individual parameters
        
        Args:
            criteria: Dictionary of filter criteria
            filter_by_start_date: Whether to filter by start date
            client_side_name_contains: String to search for in cycle names (client-side filtering)
            **kwargs: Additional filter parameters
        """
        self._check_client()
        cycles = self.client.filterCycles(criteria, filter_by_start_date)
        
        # Apply client-side name filtering if specified
        if client_side_name_contains:
            filtered_cycles = []
            for cycle in cycles:
                if cycle.get('name') and client_side_name_contains.lower() in cycle.get('name').lower():
                    filtered_cycles.append(cycle)
            return filtered_cycles
            
        return cycles
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_comments)
    def filterComments(self, criteria: dict = None, **kwargs) -> List[dict]:
        """Filter comments by criteria or individual parameters"""
        self._check_client()
        return self.client.filterComments(criteria)
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_attachments)
    def filterAttachments(self, criteria: dict = None, **kwargs) -> List[dict]:
        """Filter attachments by criteria or individual parameters"""
        self._check_client()
        return self.client.filterAttachments(criteria)
    
    # Action methods
    @adapt_parameters(LinearParameterAdapter.adapt_create_issue)
    def createIssue(self, data: dict = None, **kwargs) -> dict:
        """Create a new issue"""
        self._check_client()
        return self.client.createIssue(data)
    
    @adapt_parameters(LinearParameterAdapter.adapt_update_issue)
    def updateIssue(self, issue_id: str = None, issueId: str = None, issueNumber: int = None, data: dict = None, **kwargs) -> dict:
        """Update an existing issue"""
        self._check_client()
        
        # Check if we got an issue ID (preferred) or need to use issue number
        if issueId:
            issue_id = issueId
            
        # Get team key from kwargs or use the one in data
        team_key = kwargs.get('teamKey') or kwargs.get('team_key')
        
        # IMPORTANT: Get critical values from kwargs for resolution
        # These come from the original parameters before adaptation
        state_name = kwargs.get('state_name')
        cycle_number = kwargs.get('cycle_number')
        assignee_name = kwargs.get('assignee_name')
        parent_issue_number = kwargs.get('parent_issue_number')
        project_name = kwargs.get('project_name')
        
        # Log what we found in kwargs
        logger.info(f"Critical values from kwargs: state_name={state_name}, cycle_number={cycle_number}, " + 
                   f"assignee_name={assignee_name}, parent_issue_number={parent_issue_number}, project_name={project_name}")
        
        # Resolve placeholders in data to actual UUIDs
        if data:
            # Log the initial data before any resolution
            logger.info(f"Initial data before resolution: {data}")
            
            # Check which placeholders we have
            if 'assigneeId' in data and data['assigneeId'] == '$getUserIdByName':
                logger.info(f"Found assigneeId placeholder")
                
            if 'stateId' in data and data['stateId'] == '$getStateIdByName':
                logger.info(f"Found stateId placeholder")
                
            if 'labelIds' in data and isinstance(data['labelIds'], list) and data['labelIds'] and data['labelIds'][0] == '$getLabelIdsByNames':
                logger.info(f"Found labelIds placeholder")
                
            if 'cycleId' in data and data['cycleId'] == '$getCycleIdByNumber':
                logger.info(f"Found cycleId placeholder")
            
            # Create a copy of the data to avoid modifying the original
            resolved_data = data.copy()
            
            # Resolve assignee_name to assigneeId
            if 'assigneeId' in resolved_data and resolved_data['assigneeId'] == '$getUserIdByName':
                if assignee_name:
                    try:
                        # Get user info
                        user = self.client.getUserByName(assignee_name, team_key)
                        if user and user.get('id'):
                            resolved_data['assigneeId'] = user.get('id')
                            logger.info(f"Resolved assignee '{assignee_name}' to ID: {user.get('id')}")
                        else:
                            # If we can't resolve the assignee, remove it to avoid an error
                            logger.warning(f"Could not resolve assignee '{assignee_name}', removing from update")
                            resolved_data.pop('assigneeId')
                    except Exception as e:
                        logger.warning(f"Error resolving assignee ID: {str(e)}")
                        resolved_data.pop('assigneeId')
                else:
                    resolved_data.pop('assigneeId')
            
            # Resolve state_name to stateId
            if 'stateId' in resolved_data and resolved_data['stateId'] == '$getStateIdByName':
                if state_name and team_key:
                    try:
                        # Get all states for the team
                        states = self.client.getAllStates(team_key)
                        state = next((s for s in states if s.get('name', '').lower() == state_name.lower()), None)
                        if state and state.get('id'):
                            resolved_data['stateId'] = state.get('id')
                            logger.info(f"Resolved state '{state_name}' to ID: {state.get('id')}")
                        else:
                            # If we can't resolve the state, log clearly and remove it
                            logger.warning(f"Could not find state with name '{state_name}' for team {team_key}, removing stateId from update")
                            resolved_data.pop('stateId')
                    except Exception as e:
                        # Log the error and keep the field out of the update
                        logger.warning(f"Error resolving state ID: {str(e)}")
                        logger.warning(f"Removing stateId from update")
                        resolved_data.pop('stateId')
                else:
                    # Log why we're removing it
                    if not state_name:
                        logger.warning("No state_name provided, removing stateId from update")
                    if not team_key:
                        logger.warning("No team_key provided, removing stateId from update")
                    resolved_data.pop('stateId')
            
            # Resolve label_names to labelIds
            if 'labelIds' in resolved_data and isinstance(resolved_data['labelIds'], list) and resolved_data['labelIds'] and resolved_data['labelIds'][0] == '$getLabelIdsByNames':
                label_names = kwargs.get('label_names')
                # Even if label_names is empty, we need to handle it properly
                if label_names is not None and team_key:
                    # If it's an empty list, just set labelIds to empty list
                    if not label_names:
                        resolved_data['labelIds'] = []
                        logger.info("Setting labelIds to empty list for empty label_names")
                    else:
                        try:
                            # Get all labels for the team
                            labels = self.client.getAllLabels(team_key)
                            label_ids = []
                            for label_name in label_names:
                                label = next((l for l in labels if l.get('name', '').lower() == label_name.lower()), None)
                                if label and label.get('id'):
                                    label_ids.append(label.get('id'))
                            
                            if label_ids:
                                resolved_data['labelIds'] = label_ids
                                logger.info(f"Resolved labels to IDs: {label_ids}")
                            else:
                                # If we can't resolve any labels, set to empty list
                                logger.warning(f"Could not resolve any labels, setting empty list")
                                resolved_data['labelIds'] = []
                        except Exception as e:
                            logger.warning(f"Error resolving label IDs: {str(e)}")
                            resolved_data['labelIds'] = []
                else:
                    resolved_data.pop('labelIds')
            
            # Resolve cycle_number to cycleId
            if 'cycleId' in resolved_data and resolved_data['cycleId'] == '$getCycleIdByNumber':
                cycle_number = kwargs.get('cycle_number')
                if cycle_number and team_key:
                    try:
                        # Get all cycles for the team
                        cycles = self.client.getAllCycles(team_key)
                        cycle = next((c for c in cycles if c.get('number') == cycle_number), None)
                        if cycle and cycle.get('id'):
                            resolved_data['cycleId'] = cycle.get('id')
                            logger.info(f"Resolved cycle #{cycle_number} to ID: {cycle.get('id')}")
                        else:
                            # If we can't resolve the cycle, log clearly and remove it
                            logger.warning(f"Could not find cycle #{cycle_number} for team {team_key}, removing cycleId from update")
                            resolved_data.pop('cycleId')
                    except Exception as e:
                        # Log the error and keep the field out of the update
                        logger.warning(f"Error resolving cycle ID: {str(e)}")
                        logger.warning(f"Removing cycleId from update")
                        resolved_data.pop('cycleId')
                else:
                    # Log why we're removing it
                    if not cycle_number:
                        logger.warning("No cycle_number provided, removing cycleId from update")
                    if not team_key:
                        logger.warning("No team_key provided, removing cycleId from update") 
                    resolved_data.pop('cycleId')
            
            # Resolve parent_issue_number to parentId
            if 'parentId' in resolved_data and resolved_data['parentId'] == '$getIssueIdByNumber':
                parent_issue_number = kwargs.get('parent_issue_number')
                if parent_issue_number and team_key:
                    try:
                        # Find the parent issue by number
                        criteria = {"number": {"eq": parent_issue_number}}
                        if team_key:
                            criteria["team"] = {"key": {"eq": team_key}}
                            
                        # Look up parent issue
                        issues = self.client.filterIssues(criteria, limit=1)
                        
                        if issues and len(issues) > 0 and issues[0].get('id'):
                            parent_id = issues[0].get('id')
                            resolved_data['parentId'] = parent_id
                            logger.info(f"Resolved parent issue #{parent_issue_number} to ID: {parent_id}")
                        else:
                            # If we can't find the parent issue, log and remove it
                            logger.warning(f"Could not find parent issue #{parent_issue_number}, removing parentId from update")
                            resolved_data.pop('parentId')
                    except Exception as e:
                        # Log the error and keep the field out of the update
                        logger.warning(f"Error resolving parent issue ID: {str(e)}")
                        logger.warning(f"Removing parentId from update")
                        resolved_data.pop('parentId')
                else:
                    # Log why we're removing it
                    if not parent_issue_number:
                        logger.warning("No parent_issue_number provided, removing parentId from update")
                    if not team_key:
                        logger.warning("No team_key provided, removing parentId from update")
                    resolved_data.pop('parentId')
                    
            # Resolve project_name to projectId
            if 'projectId' in resolved_data and resolved_data['projectId'] == '$getProjectIdByName':
                project_name = kwargs.get('project_name')
                if project_name and team_key:
                    try:
                        # Get all projects for the team
                        projects = self.client.getAllProjects(team_key)
                        project = next((p for p in projects if p.get('name', '').lower() == project_name.lower()), None)
                        
                        if project and project.get('id'):
                            resolved_data['projectId'] = project.get('id')
                            logger.info(f"Resolved project '{project_name}' to ID: {project.get('id')}")
                        else:
                            # If we can't find the project, log and remove it
                            logger.warning(f"Could not find project '{project_name}' for team {team_key}, removing projectId from update")
                            resolved_data.pop('projectId')
                    except Exception as e:
                        # Log the error and keep the field out of the update
                        logger.warning(f"Error resolving project ID: {str(e)}")
                        logger.warning(f"Removing projectId from update")
                        resolved_data.pop('projectId')
                else:
                    # Log why we're removing it
                    if not project_name:
                        logger.warning("No project_name provided, removing projectId from update")
                    if not team_key:
                        logger.warning("No team_key provided, removing projectId from update")
                    resolved_data.pop('projectId')
            
            # Use the resolved data instead of the original data
            data = resolved_data
            
            # Log final state after all resolutions
            logger.info(f"Field resolution summary:")
            for field in ['title', 'description', 'priority', 'assigneeId', 'stateId', 'labelIds', 'cycleId', 'parentId', 'projectId']:
                if field in data:
                    logger.info(f"  - {field} is present in final data")
                else:
                    logger.info(f"  - {field} was removed during resolution")
            
        if issue_id:
            # If we already have the issue ID, use it directly
            return self.client.updateIssueById(issue_id, data)
        elif issueNumber:
            # Linear API has changed and no longer allows updating by number
            # We need to first get the issue ID using the number
            try:
                # Construct filter criteria to find the issue by number
                criteria = {"number": {"eq": issueNumber}}
                if team_key:
                    criteria["team"] = {"key": {"eq": team_key}}
                
                # Find the issue
                issues = self.client.filterIssues(criteria, limit=1)
                if not issues or len(issues) == 0:
                    raise ValueError(f"Issue #{issueNumber} not found")
                
                # Get the issue ID and use it to update
                issue_id = issues[0].get('id')
                if not issue_id:
                    raise ValueError(f"Issue #{issueNumber} found but has no ID")
                
                logger.info(f"Found issue ID: {issue_id} for issue #{issueNumber}")
                
                # Add detailed logging to see exactly what we're sending to the API
                logger.info(f"Sending update to Linear API with issue_id: {issue_id}")
                if data is None:
                    logger.error(f"DATA IS NULL before API call. This will cause an error. resolved_data: {resolved_data}")
                else:
                    logger.info(f"Final data payload: {data}")
                
                # Make the API call
                result = self.client.updateIssueById(issue_id, data)
                
                # Log the result
                if result:
                    logger.info(f"Linear API update response: {result}")
                
                return result
            except Exception as e:
                logger.error(f"Error updating issue by number: {str(e)}")
                raise
        else:
            raise ValueError("Either issue_id or issueNumber must be provided")
    
    @adapt_parameters(LinearParameterAdapter.adapt_create_comment)
    def createComment(self, issueNumber: int = None, issue_number: int = None, issue_id: str = None, id: str = None, 
                    teamKey: str = None, team_key: str = None, commentData: dict = None, input: dict = None, **kwargs) -> dict:
        """
        Add a comment to an issue
        
        Args:
            issue_id: The UUID of the issue
            issueNumber: Issue number (requires team_key to resolve ID)
            issue_number: Alternative name for issueNumber
            teamKey: Team key for resolving issue ID (required if using issue_number)
            team_key: Alternative name for teamKey
            commentData: Comment data dictionary containing body
            input: Alternative name for commentData
            
        Returns:
            Created comment object
            
        Raises:
            ValueError: If required parameters are missing
        """
        self._check_client()
        import logging
        logger = logging.getLogger("linear_tools")
        
        # Handle parameters that could come from adapter
        comment_data = input or commentData or {}
        issue_id_param = id or issue_id
        
        # If issue_id is provided, use it directly
        if issue_id_param:
            return self.client.createCommentById(issue_id_param, comment_data)
            
        # If we have issue number and team key, resolve the issue ID first
        issue_num = issueNumber or issue_number
        team_k = teamKey or team_key
        
        if issue_num and team_k:
            try:
                # Search for the issue to get its ID
                criteria = {
                    "team": {"key": {"eq": team_k}},
                    "number": {"eq": issue_num}
                }
                issues = self.client.filterIssues(criteria, limit=1)
                
                if issues and len(issues) > 0:
                    resolved_issue_id = issues[0].get("id")
                    logger.info(f"Resolved issue ID {resolved_issue_id} from #{issue_num} in team {team_k}")
                    
                    # Use the resolved ID to create the comment
                    return self.client.createCommentById(resolved_issue_id, comment_data)
                else:
                    raise ValueError(f"Issue #{issue_num} not found in team {team_k}")
            except Exception as e:
                raise ValueError(f"Error resolving issue ID: {str(e)}")
        
        # Backward compatibility - try with just issue number, but log a warning
        elif issue_num:
            logger.warning(f"Creating comment for issue #{issue_num} without team key is not recommended")
            return self.client.createComment(issue_num, comment_data)
            
        # If we get here, we don't have enough information
        raise ValueError("Either issue_id or both issue_number and team_key must be provided")
    
    @adapt_parameters(LinearParameterAdapter.adapt_create_attachment)
    def createAttachment(self, issueNumber: int = None, attachmentData: dict = None, **kwargs) -> dict:
        """Attach a file or link to an issue"""
        self._check_client()
        return self.client.createAttachment(issueNumber, attachmentData)

    # RAG/Semantic search methods
    def semantic_search_linear(self, query: str, limit: int = 5, use_reranker: bool = True, 
                      candidate_pool_size: int = 20, team_key: str = None, 
                      object_type: str = None) -> List[Dict]:
        """Search Linear content semantically"""
        return semantic_search(
            query=query, 
            limit=limit, 
            use_reranker=use_reranker,
            candidate_pool_size=candidate_pool_size, 
            team_key=team_key, 
            object_type=object_type
        )

# Slack Tool Functions
class SlackTools:
    """Wrapper for Slack API tools"""
    
    def __init__(self, bot_token: str = None, user_token: str = None, context_id: str = None):
        """
        Initialize Slack tools with tokens.
        
        Args:
            bot_token: Slack bot token (defaults to SLACK_BOT_TOKEN environment variable)
            user_token: Optional Slack user token (defaults to SLACK_USER_TOKEN environment variable)
            context_id: Optional context ID for conversation history
        """
        self._context_id = context_id
        if bot_token is None:
            bot_token = os.environ.get("SLACK_BOT_TOKEN")
            
        if user_token is None:
            user_token = os.environ.get("SLACK_USER_TOKEN")
            
        if not bot_token:
            logger.warning("No Slack bot token provided. Slack functionality will be limited.")
            self.client = None
        elif SlackClient is None:
            logger.warning("SlackClient module not available. Slack functionality will be limited.")
            self.client = None
        else:
            try:
                self.client = SlackClient(bot_token=bot_token, user_token=user_token)
                logger.info("Slack client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Slack client: {str(e)}")
                self.client = None
    
    def _check_client(self):
        """Check if client is initialized"""
        if not self.client:
            raise ValueError("Slack client not initialized. Please provide valid tokens.")
    
    async def upload_file(self, file: str) -> Dict:
        """
        Upload a file to Slack.
        
        Args:
            file_path: Path to the file to upload
            channel_id: Optional channel ID to upload the file to
            title: Optional title for the file
            thread_ts: Optional thread timestamp to upload the file to
            
        Returns:
            Slack API response with file details
        """
        self._check_client()
        

        # Import context_manager
        try:
            from context_manager import context_manager
        except Exception as e:
            from app.context_manager import context_manager
        
        # Parse channel_id and thread_ts from the context_id
        # Context ID format is "channel_id:thread_ts"
        try:
            parts = self._context_id.split(':', 1)
            if len(parts) == 2:
                channel_id, thread_ts = parts
                logger.info(f"Retrieved channel_id={channel_id}, thread_ts={thread_ts} from context_id")

            else:
                # Attempt to get from context directly
                context = context_manager.get_context(self._context_id)
                if context and 'channel_id' in context and 'thread_ts' in context:
                    channel_id = context['channel_id']
                    thread_ts = context['thread_ts'].split('.')[0]
                    logger.info(f"Retrieved channel_id={channel_id}, thread_ts={thread_ts} from context data")
                else:
                    logger.error(f"Invalid context_id format: {self._context_id}")
                    return ["Error: Could not parse conversation context information"]
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {str(e)}")
            return [f"Error retrieving conversation context: {str(e)}"]
        
        # Upload the file
        try:
            from slack_sdk import WebClient
            web_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
            response = web_client.files_upload_v2(channel=channel_id, file=file, thread_ts=thread_ts)
            permalink = response.get("files", {})[0].get("permalink", "")
            post_message = web_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=permalink)
            
            logger.info(f"File uploaded successfully")
            
            # Convert SlackResponse to dict to make it JSON serializable
            response_dict = {
                "success": True,
                "message": "File uploaded to Slack",
                "file_id": response.get("file_id", ""),
                "result": dict(post_message.data) if hasattr(post_message, "data") else {"text": "Message sent"}
            }
                        
            return response_dict
            
        except Exception as e:
            logger.error(f"Error uploading file to Slack: {str(e)}")
            raise
    
    def get_current_user(self, slack_display_name: str) -> dict:
        """Get information about a specific user by their Slack display name"""
        self._check_client()
        return self.client.get_employees_data(slack_display_name)
    
    async def search_channel_history(
        self,
        channel_id: str,
        username: Optional[str] = None,
        time_range: str = "days",
        time_value: int = 7,
        message_count: int = 50
    ) -> Dict[str, Any]:
        """Search channel history with specified parameters"""
        self._check_client()
        return await self.client.search_channel_history(
            channel_id=channel_id,
            username=username,
            time_range=time_range,
            time_value=time_value,
            message_count=message_count
        )
    
    def get_user(self, display_name: str = None) -> List[Dict[str, Any]]:
        """Display all users in the company"""
        self._check_client()
        return self.client.get_current_user(display_name)
        
    async def get_conversation_context(self, max_messages: int = 10) -> List[str]:
        """
        Get conversation context for the current conversation.
        Uses the context_id to retrieve channel_id and thread_ts.
        
        Args:
            max_messages: Maximum number of messages to retrieve
            
        Returns:
            List of formatted message strings
        """
        self._check_client()
        
        # Get context_id from the instance
        if not hasattr(self, '_context_id') or not self._context_id:
            logger.error("No context_id set on SlackTools instance")
            return ["Error: No conversation context available"]
        
        # Import context_manager
        try:
            from context_manager import context_manager
        except Exception as e:
            from app.context_manager import context_manager
        
        # Parse channel_id and thread_ts from the context_id
        # Context ID format is "channel_id:thread_ts"
        try:
            parts = self._context_id.split(':', 1)
            if len(parts) == 2:
                channel_id, thread_ts = parts
                logger.info(f"Retrieved channel_id={channel_id}, thread_ts={thread_ts} from context_id")
                
                # Call the client method with the extracted parameters
                return await self.client.get_conversation_context(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    max_messages=max_messages
                )
            else:
                # Attempt to get from context directly
                context = context_manager.get_context(self._context_id)
                if context and 'channel_id' in context and 'thread_ts' in context:
                    channel_id = context['channel_id']
                    thread_ts = context['thread_ts']
                    logger.info(f"Retrieved channel_id={channel_id}, thread_ts={thread_ts} from context data")
                    
                    return await self.client.get_conversation_context(
                        channel_id=channel_id,
                        thread_ts=thread_ts,
                        max_messages=max_messages
                    )
                else:
                    logger.error(f"Invalid context_id format: {self._context_id}")
                    return ["Error: Could not parse conversation context information"]
        except Exception as e:
            logger.error(f"Error retrieving conversation context: {str(e)}")
            return [f"Error retrieving conversation context: {str(e)}"]

class WebsiteTools:
    def __init__(self):
        self.db = WebsiteDB()

    def search_website_content(self, query: str, website_type: str = None, distinct_on_url: bool = False, return_full_content: bool = False, limit: int = 5) -> List[Dict[str, Any]]:
        return self.db.search_website_content(query=query, website_type=website_type, distinct_on_url=distinct_on_url, return_full_content=return_full_content, limit=limit)
    
class GDriveTools:
    def __init__(self):
        self.gdrive_client = GoogleDriveClient()

    def search_drive_files(self, query: str, limit: int) -> List[Dict[str, Any]]:
        return self.gdrive_client.search_drive_files(query=query, limit=limit)

    def get_drive_file_content(self, file_id: str) -> str:
        return self.gdrive_client.get_drive_file_content(file_id=file_id)

class PosthogTools:
    """Tools for interacting with PostHog analytics."""
    
    def __init__(self):
        """Initialize PostHog client."""
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the PostHog client with API key from environment."""
        try:
            # Load from .env if not already loaded
            load_dotenv()
            
            # Create the PostHog client
            self.client = PosthogClient()
            logging.info("PosthogTools initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing PosthogTools: {str(e)}")
            self.client = None
    
    def _check_client(self):
        """Ensure the client is initialized."""
        if not self.client:
            self._initialize_client()
            if not self.client:
                raise ValueError("PostHog client is not initialized. Check your API credentials.")
    
    def get_dashboards(self) -> List[Dict]:
        """Get a list of all dashboards in the PostHog project."""
        self._check_client()
        return self.client.get_dashboards()
    
    def get_dashboard_by_name(self, name: str) -> Dict:
        """Find a dashboard by its name."""
        self._check_client()
        return self.client.get_dashboard_by_name(name)
    
    def get_dashboard_by_id(self, dashboard_id: str) -> Dict:
        """Get a dashboard by its ID."""
        self._check_client()
        return self.client.get_dashboard_by_id(dashboard_id)
    
    def get_dashboard_items(self, dashboard_id: str) -> List[Dict]:
        """Get all insights/charts in a dashboard."""
        self._check_client()
        return self.client.get_dashboard_items(dashboard_id)
    
    def get_insight_data(self, insight_id: str, days: int = 7) -> Dict:
        """Get data for a specific insight/chart."""
        self._check_client()
        return self.client.get_insight_data(insight_id, days)
    
    def get_dashboard_data(self, dashboard_name: str, days: int = 7) -> Dict:
        """Get all data for a dashboard including all insights."""
        self._check_client()
        return self.client.get_dashboard_data(dashboard_name, days)
    
    async def get_dashboard_screenshot(self, dashboard_id: str, context_id: str = None) -> Union[bytes, str, Dict]:
        """
        Get a screenshot of a dashboard as a PNG image and optionally upload to Slack.
        
        Args:
            dashboard_id: The ID of the dashboard to screenshot
            context_id: Optional context ID for the Slack conversation
            
        Returns:
            If upload_to_slack is True, returns a dict with Slack upload details.
            Otherwise, returns the binary image data.
        """
        self._check_client()
        
        # Get the screenshot from PostHog - this is a synchronous method
        logger.info(f"Getting dashboard screenshot for dashboard_id {dashboard_id}")
        screenshot_data = self.client.get_dashboard_screenshot(dashboard_id)
        
        # If screenshot failed, return None
        if not screenshot_data:
            logger.error("Failed to get dashboard screenshot")
            return None
        else:
            logger.info(f"Screenshot data received: {len(screenshot_data) if isinstance(screenshot_data, bytes) else 'not bytes'} bytes")
        
        # Upload to Slack
        try:
            # Import SlackTools directly
            from ops_slack.slack_tools import SlackClient
            logger.info(f"Uploading screenshot to Slack")
            
            # Create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                # If screenshot_data is bytes, write directly
                if isinstance(screenshot_data, bytes):
                    temp_file.write(screenshot_data)
                    logger.info(f"Wrote {len(screenshot_data)} bytes to temporary file")
                # If it's a string path, read the file
                elif isinstance(screenshot_data, str) and os.path.exists(screenshot_data):
                    with open(screenshot_data, "rb") as src_file:
                        data = src_file.read()
                        temp_file.write(data)
                        logger.info(f"Copied {len(data)} bytes from {screenshot_data} to temporary file")
                else:
                    logger.error(f"Invalid screenshot data type: {type(screenshot_data)}")
                    return screenshot_data
            
            # Use SlackClient directly
            slack_client = SlackClient(bot_token=os.environ.get("SLACK_BOT_TOKEN"))
            
            # Get current context ID from:
            # 1. Function parameter (preferred)
            # 2. Environment variable
            # 3. Context manager
            from context_manager import context_manager
            
            if not context_id:
                context_id = os.environ.get("CURRENT_CONTEXT_ID")
                logger.info(f"Using context_id from environment: {context_id}")
            else:
                logger.info(f"Using context_id provided to function: {context_id}")
                
            context = {}
            
            if context_id:
                logger.info(f"Found context_id: {context_id}")
                context = context_manager.get_context(context_id)
                
            # Extract channel_id and thread_ts
            channel_id = None
            thread_ts = None
            
            if context and 'channel_id' in context:
                channel_id = context.get('channel_id')
                thread_ts = context.get('thread_ts')
                logger.info(f"Retrieved from context: channel_id={channel_id}, thread_ts={thread_ts}")
            elif context_id and ':' in context_id:
                parts = context_id.split(':', 1)
                if len(parts) == 2:
                    channel_id, thread_ts = parts
                    logger.info(f"Parsed from context_id: channel_id={channel_id}, thread_ts={thread_ts}")
                    
            if not channel_id:
                logger.error("No channel_id found in context")
                raise ValueError("No channel_id found in context")
                
            # Upload the file
            logger.info(f"Uploading file {temp_path} to channel {channel_id} with thread_ts {thread_ts}")
            from slack_sdk import WebClient
            web_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
            response = web_client.files_upload_v2(
                channel=channel_id,
                file=temp_path,
                thread_ts=thread_ts,
            )
            permalink = response.get("files", {})[0].get("permalink", "")
            post_message = web_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=permalink)
            
            # Delete the temporary file
            try:
                os.unlink(temp_path)
                logger.info(f"Deleted temporary file {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")
                
            # Convert SlackResponse to dict to make it JSON serializable
            response_dict = {
                "success": True,
                "message": "Insight screenshot uploaded to Slack",
                "file_id": response.get("file_id", ""),
                "result": dict(post_message.data) if hasattr(post_message, "data") else {"text": "Message sent"}
            }
                        
            return response_dict
                
        except Exception as e:
            logger.error(f"Error uploading dashboard screenshot to Slack: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return the original result if upload fails
            return screenshot_data
    
    async def get_insight_screenshot(self, insight_id: str, context_id: str = None) -> Union[bytes, str, Dict]:
        """
        Get a screenshot of an insight as a PNG image and optionally upload to Slack.
        
        Args:
            insight_id: The ID of the insight to screenshot
            context_id: Optional context ID for the Slack conversation
            
        Returns:
            If upload_to_slack is True, returns a dict with Slack upload details.
            Otherwise, returns the binary image data.
        """
        self._check_client()
        
        # Get the screenshot from PostHog - this is a synchronous method
        logger.info(f"Getting insight screenshot for insight_id {insight_id}")
        screenshot_data = self.client.get_insight_screenshot(insight_id)
        
        # If screenshot failed, return None
        if not screenshot_data:
            logger.error("Failed to get insight screenshot")
            return None
        else:
            logger.info(f"Screenshot data received: {len(screenshot_data) if isinstance(screenshot_data, bytes) else 'not bytes'} bytes")
        
        # Upload to Slack
        try:
            # Import SlackTools directly
            from ops_slack.slack_tools import SlackClient
            logger.info(f"Uploading screenshot to Slack")
            
            # Create a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                temp_path = temp_file.name
                # If screenshot_data is bytes, write directly
                if isinstance(screenshot_data, bytes):
                    temp_file.write(screenshot_data)
                    logger.info(f"Wrote {len(screenshot_data)} bytes to temporary file")
                # If it's a string path, read the file
                elif isinstance(screenshot_data, str) and os.path.exists(screenshot_data):
                    with open(screenshot_data, "rb") as src_file:
                        data = src_file.read()
                        temp_file.write(data)
                        logger.info(f"Copied {len(data)} bytes from {screenshot_data} to temporary file")
                else:
                    logger.error(f"Invalid screenshot data type: {type(screenshot_data)}")
                    return screenshot_data
            
            # Use SlackClient directly
            slack_client = SlackClient(bot_token=os.environ.get("SLACK_BOT_TOKEN"))
            
            # Get current context ID from:
            # 1. Function parameter (preferred)
            # 2. Environment variable
            # 3. Context manager
            from context_manager import context_manager
            
            if not context_id:
                context_id = os.environ.get("CURRENT_CONTEXT_ID")
                logger.info(f"Using context_id from environment: {context_id}")
            else:
                logger.info(f"Using context_id provided to function: {context_id}")
                
            context = {}
            
            if context_id:
                logger.info(f"Found context_id: {context_id}")
                context = context_manager.get_context(context_id)
                
            # Extract channel_id and thread_ts
            channel_id = None
            thread_ts = None
            
            if context and 'channel_id' in context:
                channel_id = context.get('channel_id')
                thread_ts = context.get('thread_ts')
                logger.info(f"Retrieved from context: channel_id={channel_id}, thread_ts={thread_ts}")
            elif context_id and ':' in context_id:
                parts = context_id.split(':', 1)
                if len(parts) == 2:
                    channel_id, thread_ts = parts
                    logger.info(f"Parsed from context_id: channel_id={channel_id}, thread_ts={thread_ts}")
                    
            if not channel_id:
                logger.error("No channel_id found in context")
                raise ValueError("No channel_id found in context")
                
            # Upload the file
            logger.info(f"Uploading file {temp_path} to channel {channel_id} with thread_ts {thread_ts}")
            from slack_sdk import WebClient
            web_client = WebClient(token=os.environ.get("SLACK_BOT_TOKEN"))
            response = web_client.files_upload_v2(
                channel=channel_id,
                file=temp_path,
                thread_ts=thread_ts,
                title=f"Insight: {insight_id}"
            )
            permalink = response.get("files", {})[0].get("permalink", "")
            post_message = web_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=permalink)
            
            # Delete the temporary file
            try:
                os.unlink(temp_path)
                logger.info(f"Deleted temporary file {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")
                
            # Convert SlackResponse to dict to make it JSON serializable
            response_dict = {
                "success": True,
                "message": "Insight screenshot uploaded to Slack",
                "file_id": response.get("file_id", ""),
                "result": dict(post_message.data) if hasattr(post_message, "data") else {"text": "Message sent"}
            }
                        
            return response_dict
                
        except Exception as e:
            logger.error(f"Error uploading insight screenshot to Slack: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return the original result if upload fails
            return screenshot_data
    
    def save_dashboard_screenshots(self, dashboard_name: str, output_dir: str) -> List[str]:
        """Save screenshots of a dashboard and all its insights to a directory."""
        self._check_client()
        return self.client.save_dashboard_screenshots(dashboard_name, output_dir)
    
    def generate_daily_report(self, dashboard_name: str) -> str:
        """Generate a daily analytics report for a dashboard."""
        self._check_client()
        return self.client.generate_daily_report(dashboard_name)
    
    def generate_weekly_report(self, dashboard_names: List[str]) -> str:
        """Generate a comprehensive weekly analytics report for one or more dashboards."""
        self._check_client()
        # Convert single dashboard name to list if needed
        if isinstance(dashboard_names, str):
            dashboard_names = [dashboard_names]
        return self.client.generate_weekly_report(dashboard_names)
    
    def generate_ai_insights(self, dashboard_name: str, days: int = 7, insight_type: str = "daily") -> str:
        """Generate AI-powered insights for a dashboard."""
        self._check_client()
        return self.client.generate_ai_insights(dashboard_name, days, insight_type)

# Create instances of the tool classes for importing
linear_tools = LinearTools()
slack_tools = SlackTools()
website_tools = WebsiteTools()
gdrive_tools = GDriveTools()
posthog_tools = PosthogTools()

# Export all tools
__all__ = [
    # Classes
    'LinearTools',
    'SlackTools',
    'WebsiteTools',
    'GDriveTools',
    'LinearClient',
    'SlackClient',
    
    # Exceptions
    'LinearError',
    'LinearAuthError',
    'LinearNotFoundError',
    'LinearValidationError',
    
    # Functions
    'semantic_search',
    'get_embedding',
    
    # Singletons
    'linear_tools',
    'slack_tools',
    'website_tools',
    'gdrive_tools',
    'posthog_tools',
] 