"""
Centralized tools declaration for TMAI Agent.
This module imports and exposes all available tools from different sources.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dotenv import load_dotenv

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
    def filterIssues(self, criteria: dict, limit: int = None) -> List[dict]:
        """Filter issues by criteria"""
        self._check_client()
        return self.client.filterIssues(criteria, limit)
    
    def filterUsers(self, criteria: dict) -> List[dict]:
        """Filter users by criteria"""
        self._check_client()
        return self.client.filterUsers(criteria)
    
    def filterProjects(self, criteria: dict) -> List[dict]:
        """Filter projects by criteria"""
        self._check_client()
        return self.client.filterProjects(criteria)
    
    def filterCycles(self, criteria: dict, filter_by_start_date: bool = True) -> List[dict]:
        """Filter cycles by criteria"""
        self._check_client()
        return self.client.filterCycles(criteria, filter_by_start_date)
    
    def filterComments(self, criteria: dict) -> List[dict]:
        """Filter comments by criteria"""
        self._check_client()
        return self.client.filterComments(criteria)
    
    def filterAttachments(self, criteria: dict) -> List[dict]:
        """Filter attachments by criteria"""
        self._check_client()
        return self.client.filterAttachments(criteria)
    
    # Action methods
    def createIssue(self, data: dict) -> dict:
        """Create a new issue"""
        self._check_client()
        return self.client.createIssue(data)
    
    def updateIssue(self, issueNumber: int, data: dict) -> dict:
        """Update an existing issue"""
        self._check_client()
        return self.client.updateIssue(issueNumber, data)
    
    def createComment(self, issueNumber: int, commentData: dict) -> dict:
        """Add a comment to an issue"""
        self._check_client()
        return self.client.createComment(issueNumber, commentData)
    
    def createAttachment(self, issueNumber: int, attachmentData: dict) -> dict:
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