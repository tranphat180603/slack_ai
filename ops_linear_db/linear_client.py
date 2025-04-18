import requests
import os
from typing import Optional, Dict, Any, List
import logging
import datetime
from datetime import timedelta, datetime
import json
import traceback
import dotenv
import asyncio
from openai import OpenAI
from gql.transport.exceptions import TransportQueryError

dotenv.load_dotenv()

logger = logging.getLogger("linear_client")

# Disable the noisy gql.transport.requests logger
gql_logger = logging.getLogger("gql.transport.requests")
gql_logger.setLevel(logging.WARNING)

from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

class LinearError(Exception):
    """Base exception class for Linear API errors."""
    pass

class LinearAuthError(LinearError):
    """Raised when there are authentication issues."""
    pass

class LinearNotFoundError(LinearError):
    """Raised when a requested resource is not found."""
    pass

class LinearValidationError(LinearError):
    """Raised when there are validation errors in the request."""
    pass

class LinearClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise LinearAuthError("API key is required")
            
        self.api_key = api_key
        try:
            self.client = Client(
                transport=RequestsHTTPTransport(
                    url="https://api.linear.app/graphql",
                    headers={"Authorization": f"{api_key}"},
                    use_json=True,
                ),
                fetch_schema_from_transport=True,
            )
        except Exception as e:
            raise LinearAuthError(f"Failed to initialize Linear client: {str(e)}")
    
    # ---------------------------
    # Core Data Retrieval Functions (Team-Scoped)
    # ---------------------------
    
    def _execute_query(self, query: str, variables: dict = None) -> dict:
        """
        Execute a GraphQL query with error handling.
        
        Args:
            query: The GraphQL query string
            variables: Query variables
            
        Returns:
            The query result
            
        Raises:
            LinearAuthError: When authentication fails
            LinearNotFoundError: When requested resource is not found
            LinearValidationError: When the query has validation errors
            LinearError: For other Linear API errors
        """
        try:
            return self.client.execute(query, variable_values=variables)
        except TransportQueryError as e:
            error_message = str(e)
            if "AUTHENTICATION_ERROR" in error_message:
                raise LinearAuthError("Authentication failed. Please check your API key.")
            elif "NOT_FOUND" in error_message:
                raise LinearNotFoundError(f"Resource not found: {error_message}")
            elif "BAD_USER_INPUT" in error_message:
                raise LinearValidationError(f"Invalid input: {error_message}")
            else:
                raise LinearError(f"Linear API error: {error_message}")
        except Exception as e:
            raise LinearError(f"Unexpected error: {str(e)}")
        
    def getAllTeams(self) -> List:
        return [
            {
                "key": "ENG",
                "name": "Engineering"
            },
            {
                "key": "OPS",
                "name": "Operations"
            },
            {
                "key": "MKT",
                "name": "Marketing"
            },
            {
                "key": "PRO",
                "name": "Product"
            },
            {
                "key": "RES",
                "name": "Research"
            },
            {
                "key": "AI",
                "name": "AI"
            }
        ]
        
    def getCurrentUser(self, slack_display_name: str = None) -> dict:
        users_info = {
            '@Talha': {'linear_display_name': 'talha', 'team': 'MKT'},
            '@Val': {'linear_display_name': 'val', 'team': 'PRO'},
            '@Ian Balina': {'linear_display_name': 'ian', 'team': None},
            '@Harsh': {'linear_display_name': 'harshg', 'team': 'ENG'},
            '@Andrew Tran': {'linear_display_name': 'andrew', 'team': 'AI'},
            '@Ayush Jalan': {'linear_display_name': 'ayush', 'team': 'ENG'},
            '@Drich': {'linear_display_name': 'raldrich', 'team': 'PRO'},
            '@Bartosz': {'linear_display_name': 'bartosz', 'team': 'ENG'},
            '@Jake': {'linear_display_name': 'jake', 'team': 'AI'},
            '@Roshan Ganesh': {'linear_display_name': 'roshan1', 'team': 'MKT'},
            '@Sam Monac': {'linear_display_name': 'sam', 'team': None},
            '@Favour': {'linear_display_name': 'favour', 'team': 'OPS'},
            '@Suleman Tariq': {'linear_display_name': 'suleman', 'team': 'ENG'},
            '@Zaiying Li': {'linear_display_name': 'zaiying', 'team': 'OPS'},
            '@Hemank': {'linear_display_name': 'hemank', 'team': 'RES'},
            '@Ben': {'linear_display_name': 'ben', 'team': 'PRO'},
            '@Chao': {'linear_display_name': 'chao', 'team': 'AI'},
            '@Abdullah': {'linear_display_name': 'abdullah', 'team': 'RES'},
            '@Manav': {'linear_display_name': 'manav', 'team': 'RES'},
            '@Vasilis': {'linear_display_name': 'vasilis', 'team': 'AI'},
            '@Olaitan Akintunde': {'linear_display_name': 'olaitan', 'team': 'MKT'},
            '@Chetan Kale': {'linear_display_name': 'chetan', 'team': 'RES'},
            '@ayo': {'linear_display_name': 'ayo', 'team': 'PRO'},
            '@Özcan İlhan': {'linear_display_name': 'ozcan', 'team': 'ENG'},
            '@Faith Oladejo': {'linear_display_name': 'faith', 'team': 'PRO'},
            '@Taf': {'linear_display_name': 'tafcirm', 'team': 'MKT'},
            '@Caleb N': {'linear_display_name': 'caleb', 'team': 'MKT'},
            '@divine': {'linear_display_name': 'divine', 'team': 'ENG'},
            '@Williams': {'linear_display_name': 'williams', 'team': 'ENG'},
            '@Anki Truong': {'linear_display_name': 'ankit', 'team': 'ENG'},
            '@Ryan': {'linear_display_name': 'ryan', 'team': 'MKT'},
            '@Phat': {'linear_display_name': 'phat', 'team': 'OPS'},
            '@AhmedHamdy': {'linear_display_name': 'ahmedhamdy', 'team': 'AI'},
            '@Grady': {'linear_display_name': 'grady', 'team': 'AI'},
            '@Khadijah': {'linear_display_name': 'khadijah', 'team': 'OPS'},
            '@Talha Cagri': {'linear_display_name': 'talhacagri', 'team': 'AI'},
            '@Agustín Gamoneda': {'linear_display_name': 'agustin', 'team': 'MKT'},
            '@Peterson': {'linear_display_name': 'peterson', 'team': 'ENG'}
        }
        if slack_display_name:
            return users_info.get(slack_display_name, None)
        else:
            return users_info

    def getAllUsers(self, teamKey: str) -> List[dict]:
        """
        Fetch all users for a given team.
        
        Args:
            teamKey: The team's key (e.g., 'ENG', 'OPS')
            
        Returns:
            List of active team members
            
        Raises:
            LinearNotFoundError: When team is not found
            LinearError: For other Linear API errors
        """
        if not teamKey:
            raise LinearValidationError("Team key is required")
            
        query = gql("""
        query ($teamKey: String!) {
          teams(filter: { key: { eq: $teamKey } }) {
            nodes {
              id
              name
              key
              members {
                nodes {
                  displayName
                  active
                }
              }
            }
          }
        }
        """)
        
        result = self._execute_query(query, {"teamKey": teamKey})
        teams = result.get("teams", {}).get("nodes", [])
        
        if not teams:
            raise LinearNotFoundError(f"Team '{teamKey}' not found")
            
        team_members = teams[0].get("members", {}).get("nodes", [])
        # Only filter by active status if the field exists, otherwise return all members  
        return [user for user in team_members if user.get("active", True)]
    
    def getAllIssues(self, teamKey: str, limit: int = 3, filters: dict = None):
        """
        Fetch all issues for a given team.
        Args:
            teamKey: The team's key
            limit: Maximum number of issues to return
            filters: Optional additional filters (like state, priority, estimate)
        """
        query = gql("""
        query ($teamKey: String!, $first: Int) {
          issues(
            first: $first,
            filter: { team: { key: { eq: $teamKey } } }
          ) {
            nodes {
              id
              number
              title
              description
              priority
              estimate
              cycle {
                id
                number
              }
              state {
                id
                name
                type
              }
              assignee {
                id
                displayName
              }
              project {
                id
                name
              }
              team {
                id
                key
              }
              labels {
                nodes {
                  id
                  name
                  color
                }
              }
              parent {
                id
                number
                title
              }
              children {
                nodes {
                  id
                  number
                  title
                }
              }
            }
          }
        }
        """)
        variables = {"teamKey": teamKey}
        if limit is not None:
            variables["first"] = limit
            
        result = self._execute_query(query, variables)
        issues = result.get("issues", {}).get("nodes", [])
        
        # Apply additional filters in Python if provided
        if filters:
            filtered_issues = []
            for issue in issues:
                matches = True
                for key, value in filters.items():
                    if key in issue and issue[key] != value:
                        matches = False
                        break
                if matches:
                    filtered_issues.append(issue)
            return filtered_issues
        return issues
    
    def getAllProjects(self, teamKey: str):
        """Retrieve all projects for a given team."""
        query = gql("""
        query ($teamKey: String!) {
          teams(filter: { key: { eq: $teamKey } }) {
            nodes {
              id
              projects {
                nodes {
                  id
                  name
                  description
                  state
                  priority
                  targetDate
                  progress
                  lead {
                    id
                    displayName
                  }
                }
              }
            }
          }
        }
        """)
        variables = {"teamKey": teamKey}
        result = self._execute_query(query, variables)
        teams = result.get("teams", {}).get("nodes", [])
        if teams:
            return teams[0].get("projects", {}).get("nodes", [])
        return []
    
    def getAllCycles(self, teamKey: str, filter_by_start_date: bool = True, limit: int = 10):
        """
        Retrieve all cycles for a given team.
        
        Args:
            teamKey: The team's key
            filter_by_start_date: If True, only return cycles that have started (start date <= today)
        """
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # Build the filter based on parameters
        filter_obj = {"team": {"key": {"eq": teamKey}}}
        
        # Add start date filter if requested
        if filter_by_start_date:
            filter_obj["startsAt"] = {"lte": today_date}
        
        query = gql("""
        query ($filter: CycleFilter, $first: Int) {
          cycles(filter: $filter, first: $first) {
            nodes {
              id
              number
              startsAt
              endsAt
              team {
                id
                key
              }
            }
          }
        }
        """)
        variables = {"filter": filter_obj}
        if limit is not None:
            variables["first"] = limit
            
        result = self._execute_query(query, variables)
        return result.get("cycles", {}).get("nodes", [])
    
    def getAllLabels(self, teamKey: str):
        """Retrieve all labels for a given team."""
        query = gql("""
        query ($teamKey: String!) {
          teams(filter: { key: { eq: $teamKey } }) {
            nodes {
              id
              labels {
                nodes {
                  id
                  name
                  team {
                    id
                    key
                  }
                }
              }
            }
          }
        }
        """)
        variables = {"teamKey": teamKey}
        result = self._execute_query(query, variables)
        teams = result.get("teams", {}).get("nodes", [])
        if teams:
            return teams[0].get("labels", {}).get("nodes", [])
        return []
    
    def getAllStates(self, teamKey: str):
        """Retrieve all workflow states for a given team."""
        query = gql("""
        query ($teamKey: String!) {
          teams(filter: { key: { eq: $teamKey } }) {
            nodes {
              id
              states {
                nodes {
                  id
                  name
                }
              }
            }
          }
        }
        """)
        variables = {"teamKey": teamKey}
        result = self._execute_query(query, variables)
        teams = result.get("teams", {}).get("nodes", [])
        if teams:
            return teams[0].get("states", {}).get("nodes", [])
        return []
    
    # ---------------------------
    # Detailed & Filtered Data Functions
    # ---------------------------
    
    def filterIssues(self, criteria: dict, limit: int = None, include_description: bool = True):
        """
        Return issues matching a generic criteria object.
        Example criteria:
          {
            "team": { "key": { "eq": "team123" } },
            "state": { "name": { "eq": "In Progress" } },
            "priority": { "eq": "high" }
          }
        
        Args:
            criteria: Dictionary of filter criteria
            limit: Maximum number of issues to return
            include_description: If False, strip description field from results
        """
        # Define the query fields based on include_description parameter
        description_field = "description" if include_description else ""
        
        query_string = f"""
        query ($filter: IssueFilter, $first: Int) {{
          issues(
            filter: $filter,
            first: $first
          ) {{
            nodes {{
              id
              number
              title
              {description_field}
              priority
              estimate
              cycle {{
                id
                number
              }}
              state {{
                id
                name
                type
              }}
              assignee {{
                id
                displayName
              }}
              project {{
                id
                name
              }}
              team {{
                id
                key
              }}
              labels {{
                nodes {{
                  id
                  name
                  color
                }}
              }}
              parent {{
                id
                number
                title
              }}
              children {{
                nodes {{
                  id
                  number
                  title
                }}
              }}
            }}
          }}
        }}
        """
        
        query = gql(query_string)
        variables = {"filter": criteria}
        if limit is not None:
            variables["first"] = limit
            
        result = self._execute_query(query, variables)
        issues = result.get("issues", {}).get("nodes", [])
        
        # No need to strip descriptions since we didn't request them
        # If include_description is False
        
        return issues
    
    def filterUsers(self, criteria: dict):
        """Return users matching the provided criteria."""
        query = gql("""
        query ($filter: UserFilter) {
          users(filter: $filter) {
            nodes {
              id
              displayName
              email
              active
            }
          }
        }
        """)
        variables = {"filter": criteria}
        result = self._execute_query(query, variables)
        return result.get("users", {}).get("nodes", [])
    
    def getUserByName(self, display_name: str, team_key: str = None):
        """
        Find a user by display name with their complete details including ID.
        
        Args:
            display_name: The display name to search for (case insensitive)
            team_key: Optional team key to narrow down the search
            
        Returns:
            User object with id, displayName, and active status or None if not found
        """
        query = gql("""
        query {
          users(first: 100) {
            nodes {
              id
              displayName
              email
              active
              teams {
                nodes {
                  key
                }
              }
            }
          }
        }
        """)
        
        result = self._execute_query(query, {})
        users = result.get("users", {}).get("nodes", [])
        
        # Create a normalized version of the display name for case-insensitive comparison
        normalized_name = display_name.lower().strip()
        
        # Filter by display name
        matching_users = [
            u for u in users 
            if u.get("displayName") and normalized_name == u.get("displayName").lower().strip()
        ]
        
        # If team_key is provided, filter by team membership
        if team_key and matching_users:
            team_users = []
            for user in matching_users:
                # Check if user belongs to the specified team
                teams = user.get("teams", {}).get("nodes", [])
                if any(t.get("key") == team_key for t in teams):
                    team_users.append(user)
            
            # If any users match both name and team, return the first one
            if team_users:
                return team_users[0]
        
        # If we found matches and no team filtering was applied or no team-specific matches
        if matching_users:
            return matching_users[0]
            
        return None
    
    def filterProjects(self, criteria: dict):
        """Return projects matching the provided criteria."""
        query = gql("""
        query ($filter: ProjectFilter) {
          projects(filter: $filter) {
            nodes {
              id
              name
              description
              state
              priority
              targetDate
              progress
              lead {
                id
                displayName
              }
              teams {
                nodes {
                  id
                  key
                  name
                }
              }
              issues {
                nodes {
                  id
                  number
                  title
                }
              }
            }
          }
        }
        """)
        variables = {"filter": criteria}
        result = self._execute_query(query, variables)
        
        # Process the results to create a standard format for team info
        projects = result.get("projects", {}).get("nodes", [])
        
        # For each project, extract the first team (if any) and add it directly to the project object
        for project in projects:
            teams = project.get("teams", {}).get("nodes", [])
            if teams:
                project["team"] = teams[0]  # Use the first team
            else:
                project["team"] = {"id": "", "key": "", "name": ""}
        
        return projects
    
    def filterCycles(self, criteria: dict, filter_by_start_date: bool = True):
        """
        Return cycles matching the provided criteria.
        
        Args:
            criteria: Dictionary of filter criteria
            filter_by_start_date: If True, only return cycles that have started (start date <= today)
        """
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        # Add start date filter if requested
        if filter_by_start_date:
            if "startsAt" not in criteria:
                criteria["startsAt"] = {}
            if "lte" not in criteria["startsAt"]:
                criteria["startsAt"]["lte"] = today_date
        
        query = gql("""
        query ($filter: CycleFilter) {
          cycles(filter: $filter) {
            nodes {
              id
              number
              startsAt
              endsAt
              team {
                id
                key
              }
            }
          }
        }
        """)
        variables = {"filter": criteria}
        result = self._execute_query(query, variables)
        return result.get("cycles", {}).get("nodes", [])
    
    def filterComments(self, criteria: dict):
        """Fetch all comments attached to a specific issue."""
        query = gql("""
        query ($filter: CommentFilter) {
          comments(filter: $filter) {
            nodes {
              id
              body
              user {
                id
                displayName
              }
              issue {
                id
                number
              }
            }
          }
        }
        """)
        variables = {"filter": criteria}
        result = self._execute_query(query, variables)
        return result.get("comments", {}).get("nodes", [])
    
    def filterAttachments(self, criteria: dict):
        """Retrieve all attachments, then filter locally for a specific issue if needed."""
        query = gql("""
        query {
          attachments(first: 100) {
            nodes {
              id
              title
              url
              creator {
                id
                displayName
              }
              issue {
                id
                number
              }
            }
          }
        }
        """)
        
        result = self._execute_query(query, {})
        attachments = result.get("attachments", {}).get("nodes", [])
        
        # If there's an issue_number criteria in the original parameters, filter locally
        if 'issue_number' in criteria:
            issue_number = criteria.get('issue_number')
            attachments = [a for a in attachments if a.get('issue') and a.get('issue').get('number') == issue_number]
            
        # Filter by title if specified
        if 'title' in criteria and 'contains' in criteria['title']:
            title_filter = criteria['title']['contains'].lower()
            attachments = [a for a in attachments if title_filter in a.get('title', '').lower()]
            
        # Filter by creator if specified
        if 'creator' in criteria and 'displayName' in criteria['creator'] and 'eq' in criteria['creator']['displayName']:
            creator_filter = criteria['creator']['displayName']['eq']
            attachments = [a for a in attachments if a.get('creator') and a.get('creator').get('displayName') == creator_filter]
            
        return attachments
    
    # ---------------------------
    # Action Functions
    # ---------------------------
    
    def createIssue(self, data: dict) -> dict:
        """
        Create a new issue with the provided details.
        
        Args:
            data: Dictionary containing issue fields:
                - teamId (required): The ID of the team
                - title (required): Issue title
                - description: Markdown description
                - priority: Float (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)
                - estimate: Float value for estimate points
                - stateId: ID of the workflow state
                - assigneeId: ID of the user to assign to
                - labelIds: List of label IDs to assign
                - cycleId: ID of the cycle
                - projectId: ID of the project
                - parentId: ID of the parent issue
                
        Returns:
            Created issue data
            
        Raises:
            LinearValidationError: When required fields are missing
            LinearError: For other Linear API errors
        """
        required_fields = ["teamId", "title"]
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise LinearValidationError(f"Missing required fields: {', '.join(missing_fields)}")
            
        if "priority" in data and not isinstance(data["priority"], (int, float)):
            raise LinearValidationError("Priority must be a number")
            
        query = gql("""
        mutation ($input: IssueCreateInput!) {
          issueCreate(input: $input) {
            success
            issue {
              id
              number
              title
              description
              priority
              estimate
              state {
                id
                name
                type
              }
              assignee {
                id
                displayName
              }
              labels {
                nodes {
                  id
                  name
                }
              }
              cycle {
                id
                number
              }
              project {
                id
                name
              }
              team {
                id
                key
              }
              parent {
                id
                number
              }
            }
          }
        }
        """)
        
        result = self._execute_query(query, {"input": data})
        issue_result = result.get("issueCreate", {})
        
        if not issue_result.get("success"):
            raise LinearError("Failed to create issue")
            
        return issue_result

    def updateIssue(self, issueNumber: int, data: dict):
        """
        Update an existing issue identified by issueNumber.
        
        Args:
            issueNumber: The issue number to update
            data: Dictionary containing fields to update:
                - title: New title
                - description: New markdown description
                - priority: Float (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)
                - estimate: Float value for estimate points
                - stateId: ID of the new workflow state
                - assigneeId: ID of the user to assign to
                - labelIds: List of label IDs to assign
                - cycleId: ID of the cycle to move to
                - projectId: ID of the project to move to
                - parentId: ID of the new parent issue
                - archived: Boolean to archive/unarchive the issue
        """
        query = gql("""
        mutation ($number: Int!, $input: IssueUpdateInput!) {
          issueUpdate(number: $number, input: $input) {
            success
            issue {
              id
              number
              title
              description
              priority
              estimate
              state {
                id
                name
                type
              }
              assignee {
                id
                displayName
              }
              labels {
                nodes {
                  id
                  name
                }
              }
              cycle {
                id
                number
              }
              project {
                id
                name
              }
              team {
                id
                key
              }
              parent {
                id
                number
              }
              archivedAt
            }
          }
        }
        """)
        variables = {"number": issueNumber, "input": data}
        result = self._execute_query(query, variables)
        return result.get("issueUpdate", {})
        
    def updateIssueById(self, issueId: str, data: dict):
        """
        Update an existing issue identified by issueId (UUID).
        
        Args:
            issueId: The issue ID (UUID) to update
            data: Dictionary containing fields to update:
                - title: New title
                - description: New markdown description
                - priority: Float (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)
                - estimate: Float value for estimate points
                - stateId: ID of the new workflow state
                - assigneeId: ID of the user to assign to
                - labelIds: List of label IDs to assign
                - cycleId: ID of the cycle to move to
                - projectId: ID of the project to move to
                - parentId: ID of the new parent issue
                - archived: Boolean to archive/unarchive the issue
        """
        query = gql("""
        mutation ($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) {
            success
            issue {
              id
              number
              title
              description
              priority
              estimate
              state {
                id
                name
                type
              }
              assignee {
                id
                displayName
              }
              labels {
                nodes {
                  id
                  name
                }
              }
              cycle {
                id
                number
              }
              project {
                id
                name
              }
              team {
                id
                key
              }
              parent {
                id
                number
              }
              archivedAt
            }
          }
        }
        """)
        variables = {"id": issueId, "input": data}
        result = self._execute_query(query, variables)
        return result.get("issueUpdate", {}).get("issue", {})

    def createComment(self, issueNumber: int, commentData: dict):
        """Add a comment to an issue."""
        # First get the issue ID from the issue number by using filterIssues
        try:
            issue_criteria = {"number": {"eq": issueNumber}}
            issues = self.filterIssues(issue_criteria)
            
            if not issues or len(issues) == 0:
                raise LinearNotFoundError(f"Issue with number {issueNumber} not found")
                
            issueId = issues[0].get("id")
            if not issueId:
                raise LinearError(f"Issue with number {issueNumber} found but has no ID")
                
            # Now create comment using the issue ID
            return self.createCommentById(issueId, commentData)
            
        except Exception as e:
            raise LinearError(f"Error creating comment: {str(e)}")

    def createCommentById(self, issueId: str, commentData: dict):
        """Add a comment to an issue using the issue ID instead of number."""
        # The Linear API expects the issueId inside the input object
        if not isinstance(commentData, dict):
            commentData = {}
        
        # Add issueId to the input
        commentData["issueId"] = issueId
        
        query = gql("""
        mutation ($input: CommentCreateInput!) {
          commentCreate(input: $input) {
            comment {
              id
              body
              user {
                id
                displayName
              }
            }
          }
        }
        """)
        variables = {"input": commentData}
        result = self._execute_query(query, variables)
        return result.get("commentCreate", {}).get("comment", {})
    
    def createAttachment(self, issueNumber: int, attachmentData: dict):
        """Attach a file or link to an issue."""
        query = gql("""
        mutation ($issueNumber: Int!, $input: AttachmentCreateInput!) {
          attachmentCreate(issueNumber: $issueNumber, input: $input) {
            attachment {
              id
              number
              title
              url
              creator {
                id
                displayName
              }
            }
          }
        }
        """)
        variables = {"issueNumber": issueNumber, "input": attachmentData}
        result = self._execute_query(query, variables)
        return result.get("attachmentCreate", {}).get("attachment", {})

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    TEAM_KEY = "OPS"
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    linear = LinearClient(os.getenv('LINEAR_API_KEY'))
    
    
    # Get all users for a team
    users = linear.getAllUsers(TEAM_KEY)
    print("Users:", users)
    
    # Get all issues for a team (optionally with extra filters)
    issues = linear.getAllIssues(TEAM_KEY)
    print("Issues:", issues)

    projects = linear.getAllProjects(TEAM_KEY)
    print("Projects:", projects)

    cycles = linear.getAllCycles(TEAM_KEY)
    print("Cycles:", cycles)
    
    # Example filter criteria for issues:
    issue_filter = {
        "team": { "key": { "eq": TEAM_KEY } },
        "state": { "name": { "eq": "In Progress" } },
        "priority": { "eq": 1 }
    }
    filtered_issues = linear.filterIssues(issue_filter)
    print("Filtered Issues:", filtered_issues)
    
    # Test filtering projects by name
    project_filter = {
        "name": { "eq": "Improve Lucky X Agent" }
    }
    filtered_projects = linear.filterProjects(project_filter)
    print("\nFiltered Projects by name 'Improve Lucky X Agent':", filtered_projects)
    
    # Get all labels and states to let the LLM know what is available
    labels = linear.getAllLabels(TEAM_KEY)
    states = linear.getAllStates(TEAM_KEY)
    print("Labels:", labels)
    print("States:", states)