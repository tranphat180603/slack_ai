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
        Adapts parameters from the OpenAI function schema to the criteria needed for filterIssues.
        
        Only adds parameters to the criteria if they have non-null, non-empty values.
        """
        criteria = {}
        if "criteria" in params:
            return params
        
        # Team parameter is typically needed for Linear operations
        if params.get("teamKey") and params["teamKey"].strip():
            criteria["team"] = {"key": {"eq": params["teamKey"]}}
        
        # Add state filter if provided
        if params.get("state") and params["state"].strip():
            criteria["state"] = {"name": {"eq": params["state"]}}
        
        # Add priority filter if provided and not zero
        # Note: In Linear, priority 0 is valid but we'll skip it if it's exactly 0.0
        if params.get("priority") is not None and params["priority"] != 0 and params["priority"] != 0.0:
            criteria["priority"] = {"eq": float(params["priority"])}
        
        # Add assignee filter if provided
        if params.get("assignee_name") and params["assignee_name"].strip():
            criteria["assignee"] = {"displayName": {"eq": params["assignee_name"]}}
        
        # Add cycle filter if provided
        if params.get("cycle_number") is not None and params["cycle_number"] != 0:
            criteria["cycle"] = {"number": {"eq": params["cycle_number"]}}
        
        # Add project filter if provided
        if params.get("project_id") and params["project_id"].strip():
            criteria["project"] = {"id": {"eq": params["project_id"]}}
        
        # Add label filter if provided
        if params.get("label_name") and params["label_name"].strip():
            # For label, we use a different structure that finds issues with ANY label matching the name
            criteria["labels"] = {"some": {"name": {"eq": params["label_name"]}}}
        
        result = {"criteria": criteria}
        
        # Add limit if provided
        if params.get("limit") is not None and params["limit"] > 0:
            result["limit"] = params["limit"]
        
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
            
        if 'email' in params and params['email'] is not None and params['email'] != '':
            criteria['email'] = {'eq': params['email']}
            
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
        if 'teamKey' in params and params['teamKey'] is not None and params['teamKey'] != '':
            criteria['team'] = {'key': {'eq': params['teamKey']}}
            
        if 'name' in params and params['name'] is not None and params['name'] != '':
            criteria['name'] = {'eq': params['name']}
        elif 'name_contains' in params and params['name_contains'] is not None and params['name_contains'] != '':
            criteria['name'] = {'contains': params['name_contains']}
            
        if 'state' in params and params['state'] is not None and params['state'] != '':
            criteria['state'] = {'eq': params['state']}
            
        # Adjust lead_name to match the field in GraphQL query (lead.displayName)
        if 'lead_display_name' in params and params['lead_display_name'] is not None and params['lead_display_name'] != '':
            criteria['lead'] = {'displayName': {'eq': params['lead_display_name']}}
        
        return {'criteria': criteria}
    
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
            
        # Handle date filters which are in the GraphQL query
        if 'starts_at' in params and params['starts_at'] is not None and params['starts_at'] != '':
            criteria['startsAt'] = {'eq': params['starts_at']}
        
        if 'ends_at' in params and params['ends_at'] is not None and params['ends_at'] != '':
            criteria['endsAt'] = {'eq': params['ends_at']}
        
        # Get filter_by_start_date but don't add it to criteria
        filter_by_start_date = params.get('filter_by_start_date')
        if filter_by_start_date is None:  # If not specified, use default True
            filter_by_start_date = True
        
        # Preserve filter_by_start_date parameter
        return {
            'criteria': criteria,
            'filter_by_start_date': filter_by_start_date
        }
    
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
            criteria['body'] = {'contains': params['body_contains']}
            
        # User displayName is the correct field in GraphQL
        if 'user_display_name' in params and params['user_display_name'] is not None and params['user_display_name'] != '':
            criteria['user'] = {'displayName': {'eq': params['user_display_name']}}
        
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
            criteria['title'] = {'contains': params['title_contains']}
            
        # Creator displayName will be filtered client-side
        if 'creator_display_name' in params and params['creator_display_name'] is not None and params['creator_display_name'] != '':
            criteria['creator'] = {'displayName': {'eq': params['creator_display_name']}}
        
        return {'criteria': criteria}
    
    @staticmethod
    def adapt_create_issue(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat createIssue parameters to data object"""
        if 'data' in params and params['data'] is not None:
            return params
            
        # Extract the issue data from flat parameters
        data = {}
        
        # Map direct field copies
        for field in ['teamId', 'title', 'description', 'priority', 'estimate',
                     'stateId', 'assigneeId', 'labelIds', 'cycleId', 'projectId', 'parentId']:
            if field in params and params[field] is not None:
                data[field] = params[field]
                
        # Special handling for team_key if teamId is not provided
        if 'team_key' in params and 'teamId' not in data:
            # In a real implementation, we'd need to look up the teamId from the key
            pass
            
        return {'data': data}
    
    @staticmethod
    def adapt_update_issue(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat updateIssue parameters to proper format"""
        if 'data' in params and params['data'] is not None:
            return params
            
        # Extract the issue update data from flat parameters
        data = {}
        issueNumber = params.get('issueNumber') or params.get('issue_number')
        
        # Map direct field copies
        for field in ['title', 'description', 'priority', 'estimate',
                     'stateId', 'assigneeId', 'labelIds', 'cycleId', 
                     'projectId', 'parentId', 'archived']:
            if field in params and params[field] is not None:
                data[field] = params[field]
                
        return {'issueNumber': issueNumber, 'data': data}
    
    @staticmethod
    def adapt_create_comment(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat createComment parameters to proper format"""
        if 'commentData' in params and params['commentData'] is not None:
            return params
            
        issueNumber = params.get('issueNumber') or params.get('issue_number')
        commentData = {}
        
        # Extract comment body
        if 'body' in params:
            commentData['body'] = params['body']
            
        return {'issueNumber': issueNumber, 'commentData': commentData}
    
    @staticmethod
    def adapt_create_attachment(params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert flat createAttachment parameters to proper format"""
        if 'attachmentData' in params and params['attachmentData'] is not None:
            return params
            
        issueNumber = params.get('issueNumber') or params.get('issue_number')
        attachmentData = {}
        
        # Extract attachment fields
        for field in ['title', 'url']:
            if field in params and params[field] is not None:
                attachmentData[field] = params[field]
                
        return {'issueNumber': issueNumber, 'attachmentData': attachmentData}


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
                return func(self, **adapted_params)
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
        """Get information about a specific user by their Slack display name"""
        self._check_client()
        return self.client.getCurrentUser(slack_display_name)
    
    # Filter methods
    @adapt_parameters(LinearParameterAdapter.adapt_filter_issues)
    def filterIssues(self, criteria: dict = None, limit: int = None, **kwargs) -> List[dict]:
        """Filter issues by criteria or individual parameters"""
        self._check_client()
        return self.client.filterIssues(criteria, limit)
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_users)
    def filterUsers(self, criteria: dict = None, **kwargs) -> List[dict]:
        """Filter users by criteria or individual parameters"""
        self._check_client()
        return self.client.filterUsers(criteria)
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_projects)
    def filterProjects(self, criteria: dict = None, **kwargs) -> List[dict]:
        """Filter projects by criteria or individual parameters"""
        self._check_client()
        return self.client.filterProjects(criteria)
    
    @adapt_parameters(LinearParameterAdapter.adapt_filter_cycles)
    def filterCycles(self, criteria: dict = None, filter_by_start_date: bool = True, **kwargs) -> List[dict]:
        """Filter cycles by criteria or individual parameters"""
        self._check_client()
        return self.client.filterCycles(criteria, filter_by_start_date)
    
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
    def updateIssue(self, issueNumber: int = None, data: dict = None, **kwargs) -> dict:
        """Update an existing issue"""
        self._check_client()
        return self.client.updateIssue(issueNumber, data)
    
    @adapt_parameters(LinearParameterAdapter.adapt_create_comment)
    def createComment(self, issueNumber: int = None, commentData: dict = None, **kwargs) -> dict:
        """Add a comment to an issue"""
        self._check_client()
        return self.client.createComment(issueNumber, commentData)
    
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