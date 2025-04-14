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

load_dotenv()

# Configure logger
logger = logging.getLogger("tools_declaration")

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

# Import Slack tools
from ops_slack.slack_tools import SlackClient

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
        Convert flat updateIssue parameters to proper format.
        Automatically looks up IDs from names and numbers.
        """
        import logging  # Import logging at the function level to fix the error
        
        if 'data' in params and params['data'] is not None:
            return params
            
        # Extract the issue update data from flat parameters
        data = {}
        linear_client = None
        issue_id = params.get('issue_id') or params.get('issueId')
        issue_number = params.get('issueNumber') or params.get('issue_number')
        
        # Map direct field copies
        for field in ['title', 'description', 'priority', 'estimate',
                     'stateId', 'assigneeId', 'labelIds', 'cycleId', 
                     'projectId', 'parentId', 'archivedAt']:
            if field in params and params[field] is not None:
                data[field] = params[field]
        
        # Initialize LinearClient if needed for lookups
        if linear_client is None and (issue_id or issue_number or 'assignee_name' in params
                                    or 'state_name' in params or 'label_names' in params):
            try:
                from ops_linear_db.linear_client import LinearClient
                import os
                linear_client = LinearClient(os.environ.get("LINEAR_API_KEY"))
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error initializing LinearClient: {str(e)}")
                linear_client = None
        
        # Look up the issue ID from the issue number if not provided
        team_key = None
        if not issue_id and issue_number and linear_client:
            try:
                # Find the issue by number
                issue_criteria = {"number": {"eq": issue_number}}
                issues = linear_client.filterIssues(issue_criteria)
                
                if issues and len(issues) > 0:
                    issue = issues[0]
                    issue_id = issue.get('id')
                    # Extract team key for later use
                    if 'team' in issue and issue['team']:
                        team_key = issue['team'].get('key')
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up issue ID: {str(e)}")
        
        # Fetch issue details if we have the ID but not the team key
        if issue_id and not team_key and linear_client:
            try:
                # Find the issue by ID
                issue_criteria = {"id": {"eq": issue_id}}
                issues = linear_client.filterIssues(issue_criteria)
                
                if issues and len(issues) > 0:
                    issue = issues[0]
                    # Extract team key for later use
                    if 'team' in issue and issue['team']:
                        team_key = issue['team'].get('key')
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error fetching issue details: {str(e)}")
        
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
                lookup_team_key = team_key or params.get('teamKey')
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
                # Use team key from issue if available, otherwise use provided team key
                lookup_team_key = team_key or params.get('teamKey')
                
                if lookup_team_key and linear_client:
                    states = linear_client.getAllStates(lookup_team_key)
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
                # Use team key from issue if available, otherwise use provided team key
                lookup_team_key = team_key or params.get('teamKey')
                cycle_name = params.get('cycle_name')
                cycle_number = params.get('cycle_number')
                
                if lookup_team_key and linear_client:
                    cycles = linear_client.getAllCycles(lookup_team_key)
                    cycle = None
                    
                    if cycle_name:
                        cycle = next((c for c in cycles 
                                    if c.get('name', '').lower() == cycle_name.lower()), None)
                    elif cycle_number:
                        cycle = next((c for c in cycles 
                                    if c.get('number') == cycle_number), None)
                    
                    if cycle and 'id' in cycle:
                        data['cycleId'] = cycle['id']
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up cycle ID: {str(e)}")
        
        # Handle project_name to projectId conversion
        if 'project_name' in params and params['project_name'] and 'projectId' not in data:
            try:
                # Use team key from issue if available, otherwise use provided team key
                lookup_team_key = team_key or params.get('teamKey')
                
                if lookup_team_key and linear_client:
                    projects = linear_client.getAllProjects(lookup_team_key)
                    project = next((p for p in projects 
                                  if p.get('name', '').lower() == params['project_name'].lower()), None)
                    
                    if project and 'id' in project:
                        data['projectId'] = project['id']
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up project ID: {str(e)}")
        
        # Handle label_names to labelIds conversion
        if 'label_names' in params and params['label_names'] and 'labelIds' not in data:
            try:
                # Use team key from issue if available, otherwise use provided team key
                lookup_team_key = team_key or params.get('teamKey')
                
                if lookup_team_key and linear_client:
                    labels = linear_client.getAllLabels(lookup_team_key)
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
                if linear_client:
                    # Try to find the parent issue
                    issue_criteria = {"number": {"eq": params['parent_issue_number']}}
                    
                    # If we have a team key, use it to narrow down the search
                    if team_key:
                        issue_criteria["team"] = {"key": {"eq": team_key}}
                    elif 'teamKey' in params and params['teamKey']:
                        issue_criteria["team"] = {"key": {"eq": params['teamKey']}}
                        
                    issues = linear_client.filterIssues(issue_criteria)
                    
                    if issues and len(issues) > 0:
                        data['parentId'] = issues[0].get('id')
            except Exception as e:
                logging.getLogger("tools_declaration").error(f"Error looking up parent issue ID: {str(e)}")
                
        return {'issueId': issue_id, 'data': data}
    
    @staticmethod
    def adapt_create_comment(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert flat createComment parameters to proper format.
        Automatically looks up issue ID from issue number and team key.
        """
        import logging  # Import logging at the function level to fix the error
        logger = logging.getLogger("tools_declaration")
        
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
            
        if issue_id:
            return self.client.updateIssueById(issue_id, data)
        elif issueNumber:
            return self.client.updateIssue(issueNumber, data)
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
    
    def __init__(self, bot_token: str = None, user_token: str = None):
        """
        Initialize Slack tools with tokens.
        
        Args:
            bot_token: Slack bot token (defaults to SLACK_BOT_TOKEN environment variable)
            user_token: Optional Slack user token (defaults to SLACK_USER_TOKEN environment variable)
        """
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
    
    def get_users(self, display_name: str = None) -> List[Dict[str, Any]]:
        """Get employee data by display name or get all employees"""
        self._check_client()
        return self.client.get_employees_data(display_name)
    
    def extractUrls(self, text: str) -> List[str]:
        """Extract URLs from text"""
        self._check_client()
        return self.client.extract_urls(text)
    
    def formatForSlack(self, text: str) -> str:
        """Format text for Slack display"""
        self._check_client()
        return self.client.format_for_slack(text)

# Create singleton instances
linear_tools = LinearTools()
slack_tools = SlackTools()

# Export all tools
__all__ = [
    # Classes
    'LinearTools',
    'SlackTools',
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
] 

# Test functions for parameter adaptation
def run_linear_tools_tests():
    """Run tests for LinearTools methods with flat parameters (as returned by AI)"""
    print("\n=== Running LinearTools Tests with Flat Parameters ===\n")
    
    # Test data
    TEAM_KEY = "OPS"
    USER_DISPLAY_NAME = "@Phat"
    CYCLE_NUMBER = 42
    
    # Test getAllUsers
    print("\n--- Testing getAllUsers ---")
    try:
        users = linear_tools.getAllUsers(TEAM_KEY)
        print(f"Found {len(users)} users in team {TEAM_KEY}")
    except Exception as e:
        print(f"Error in getAllUsers: {str(e)}")
    
    # Test getCurrentUser
    print("\n--- Testing getCurrentUser ---")
    try:
        user = linear_tools.getCurrentUser(USER_DISPLAY_NAME)
        print(f"Found user: {user}")
    except Exception as e:
        print(f"Error in getCurrentUser: {str(e)}")
    
    # Test getAllProjects
    print("\n--- Testing getAllProjects ---")
    try:
        projects = linear_tools.getAllProjects(TEAM_KEY)
        print(f"Found {len(projects)} projects in team {TEAM_KEY}")
    except Exception as e:
        print(f"Error in getAllProjects: {str(e)}")
    
    # Test getAllCycles
    print("\n--- Testing getAllCycles ---")
    try:
        cycles = linear_tools.getAllCycles(TEAM_KEY)
        print(f"Found {len(cycles)} cycles in team {TEAM_KEY}")
    except Exception as e:
        print(f"Error in getAllCycles: {str(e)}")
    
    # Test getAllLabels
    print("\n--- Testing getAllLabels ---")
    try:
        labels = linear_tools.getAllLabels(TEAM_KEY)
        print(f"Found {len(labels)} labels in team {TEAM_KEY}")
    except Exception as e:
        print(f"Error in getAllLabels: {str(e)}")
    
    # Test getAllStates
    print("\n--- Testing getAllStates ---")
    try:
        states = linear_tools.getAllStates(TEAM_KEY)
        print(f"Found {len(states)} states in team {TEAM_KEY}")
    except Exception as e:
        print(f"Error in getAllStates: {str(e)}")
    
    # Test filterIssues with flat parameters - basic search
    print("\n--- Testing filterIssues with flat parameters (basic) ---")
    try:
        issues = linear_tools.filterIssues(team_key=TEAM_KEY, first=5)
        print(f"Found {len(issues)} issues with team_key={TEAM_KEY}")
    except Exception as e:
        print(f"Error in filterIssues (basic): {str(e)}")
        
    # Test filterIssues with flat parameters - complex search using correct fields
    print("\n--- Testing filterIssues with flat parameters (complex) ---")
    try:
        issues = linear_tools.filterIssues(
            team_key=TEAM_KEY,
            priority=3.0,  # Medium priority
            assignee_name="phat",
            title_contains="Slack",
            description_contains="Agent",
            cycle_number=CYCLE_NUMBER,  # Use cycle_number instead of cycle_name
            first=10
        )
        print(f"Found {len(issues)} issues with complex filter criteria")
    except Exception as e:
        print(f"Error in filterIssues (complex): {str(e)}")
    
    # Test filterUsers with flat parameters - active users
    print("\n--- Testing filterUsers with flat parameters ---")
    try:
        users = linear_tools.filterUsers(display_name="phat", is_active=True)
        print(f"Found {len(users)} active users matching display_name='phat'")
    except Exception as e:
        print(f"Error in filterUsers: {str(e)}")
    
    # Test filterProjects with flat parameters - using correct fields
    print("\n--- Testing filterProjects with flat parameters ---")
    try:
        projects = linear_tools.filterProjects(
            team_key=TEAM_KEY,
            name="Core",  # Project name is available in the GraphQL query
            state="started",  # Example state
            lead_display_name="phat"  # Using the correct parameter name
        )
        print(f"Found {len(projects)} projects with filter criteria")
    except Exception as e:
        print(f"Error in filterProjects: {str(e)}")
    
    # Test filterCycles with flat parameters - using correct fields
    print("\n--- Testing filterCycles with flat parameters ---")
    try:
        cycles = linear_tools.filterCycles(
            team_key=TEAM_KEY,
            number=CYCLE_NUMBER,  # Use number instead of name
            filter_by_start_date=True
        )
        print(f"Found {len(cycles)} cycles with filter criteria")
    except Exception as e:
        print(f"Error in filterCycles: {str(e)}")
    
    # Test filterComments with flat parameters - using all fields from GraphQL
    print("\n--- Testing filterComments with flat parameters ---")
    try:
        comments = linear_tools.filterComments(
            issue_number=1,  # Using a known issue number
            body_contains="test",  # Changed from 'contains' to 'body_contains'
            user_display_name="Ngoc Phat"
        )
        print(f"Found {len(comments)} comments for issue #1")
    except Exception as e:
        print(f"Error in filterComments: {str(e)}")
    
    # Test filterAttachments with flat parameters - using all fields from GraphQL
    print("\n--- Testing filterAttachments with flat parameters ---")
    try:
        attachments = linear_tools.filterAttachments(
            issue_number=1,  # Using a known issue number
            title_contains="document",
            creator_display_name="Ngoc Phat"
        )
        print(f"Found {len(attachments)} attachments for issue #1")
    except Exception as e:
        print(f"Error in filterAttachments: {str(e)}")
    
    # Test createComment with flat parameters
    print("\n--- Testing createComment with flat parameters ---")
    try:
        # This is just a test - we'll catch the exception without creating a real comment
        try:
            comment = linear_tools.createComment(
                issue_number=999999,  # Using a non-existent issue number to avoid making real changes
                body="This is a test comment"
            )
            print(f"Comment created: {comment}")
        except LinearNotFoundError:
            print("Test successful - caught expected error for non-existent issue")
        except Exception as e:
            print(f"Error in createComment: {str(e)}")
    except Exception as e:
        print(f"Error setting up createComment test: {str(e)}")
    
    # Test semantic_search_linear with flat parameters
    print("\n--- Testing semantic_search_linear with flat parameters ---")
    try:
        results = linear_tools.semantic_search_linear(
            query="Find OPS team issues about testing",
            limit=3,
            use_reranker=True,
            candidate_pool_size=10,
            team_key=TEAM_KEY,
            object_type="Issue"
        )
        print(f"Found {len(results)} semantic search results")
        print(results)
    except Exception as e:
        print(f"Error in semantic_search_linear: {str(e)}")
    
    print("\n=== LinearTools Tests Completed ===\n")

# Run tests if file is executed directly
if __name__ == "__main__":
    # Temporarily test only filterIssues with specific parameters
    print("\n=== Testing filterIssues with Specific Parameters ===\n")
    
    try:
        issues = linear_tools.filterIssues(
            team_key="OPS",
            state="Todo",
            priority=0.0,
            assignee_name="phat",
            assignee_contains="",
            title_contains="",
            description_contains="",
            cycle_number=42,
            project_id="",
            label_name="",
            first=50
        )
        print(f"Found {len(issues)} issues matching criteria")
        print("Applied filter criteria:")
        adapted = LinearParameterAdapter.adapt_filter_issues({
            'team_key': 'OPS', 
            'state': 'Todo', 
            'priority': 0.0, 
            'assignee_name': 'phat', 
            'cycle_number': 42, 
            'first': 50
        })
        print(f"  {adapted}")
        
        # Print first issue details if any found
        if issues and len(issues) > 0:
            print("\nFirst matching issue:")
            issue = issues[0]
            print(f"  Title: {issue.get('title')}")
            print(f"  State: {issue.get('state', {}).get('name')}")
            print(f"  Priority: {issue.get('priority')}")
            print(f"  Assignee: {issue.get('assignee', {}).get('displayName')}")
            if issue.get('cycle'):
                print(f"  Cycle: #{issue.get('cycle').get('number')}")
            
    except Exception as e:
        print(f"Error in filterIssues test: {str(e)}")
    
    print("\n=== FilterIssues Test Completed ===\n")
    
    # run_linear_tools_tests() # Original test function commented out 