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

dotenv.load_dotenv()

logger = logging.getLogger("linear_client")

class LinearIssue:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "")
        self.title = data.get("title", "")
        self.description = data.get("description", "")
        self.state = data.get("state", {})
        self.assignee = data.get("assignee", {})
        
        # Extract labels properly from the nodes structure
        labels_data = data.get("labels", {}).get("nodes", []) if data.get("labels") else []
        self.labels = labels_data
        
        self.priority = data.get("priority")
        self.created_at = data.get("createdAt", "")

class LinearTeam:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "")
        self.name = data.get("name", "")
        self.key = data.get("key", "")
        self.description = data.get("description", "")
        self.members_count = data.get("members", {}).get("count", 0) if data.get("members") else 0
    
    def __str__(self) -> str:
        """String representation of the team."""
        return f"Team: {self.name} (Key: {self.key})"
    
    def __repr__(self) -> str:
        """Detailed representation of the team."""
        return f"LinearTeam(name='{self.name}', key='{self.key}', members={self.members_count})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert team object to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "key": self.key,
            "description": self.description,
            "members_count": self.members_count
        }

class LinearUser:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "")
        self.name = data.get("name", "")
        self.email = data.get("email", "")
        self.display_name = data.get("displayName", "")
        self.active = data.get("active", True)
        self.admin = data.get("admin", False)
        self.teams = [team for team in data.get("teams", {}).get("nodes", [])] if data.get("teams") else []
    
    def __str__(self) -> str:
        """String representation of the user."""
        return f"User: {self.name} ({self.display_name})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user object to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "email": self.email,
            "display_name": self.display_name,
            "active": self.active,
            "admin": self.admin,
            "teams": [team.get("key") for team in self.teams] if self.teams else []
        }

class LinearWorkItem:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "")
        self.title = data.get("title", "")
        self.description = data.get("description", "")
        self.state = data.get("state", {}).get("name", "") if data.get("state") else ""
        self.creator = data.get("creator", {}).get("name", "") if data.get("creator") else ""
        self.assignee = data.get("assignee", {}).get("name", "") if data.get("assignee") else ""
        self.created_at = data.get("createdAt", "")
        self.updated_at = data.get("updatedAt", "")
        self.completed_at = data.get("completedAt", "")
        self.cycle = data.get("cycle", {}).get("name", "") if data.get("cycle") else ""
        self.estimate = data.get("estimate", 0)
        self.time_spent = data.get("timeSpent", {}).get("value", 0) if data.get("timeSpent") else 0
        self.time_logged = data.get("timeTracked", 0) if data.get("timeTracked") else 0

class LinearCycle:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "")
        self.name = data.get("name", "")
        self.number = data.get("number", 0)
        self.starts_at = data.get("startsAt", "")
        self.ends_at = data.get("endsAt", "")
        self.completed_at = data.get("completedAt", "")
        self.progress = data.get("progress", 0)
        self.scope = data.get("scope", 0)
        self.completed_scope = data.get("completedScope", 0)
        self.team = data.get("team", {})
        self.issues = data.get("issues", {}).get("nodes", []) if data.get("issues") else []

class LinearClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.linear.app/graphql"
        self.headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger("linear_client")
    
    def get_all_teams(self) -> List[LinearTeam]:
        """Get all teams in the workspace."""
        self.logger.info("Getting all teams in the workspace")
        
        try:
            query = """
            query Teams {
              teams {
                nodes {
                  id
                  name
                  key
                  description
                  members {
                    nodes {
                      id
                    }
                  }
                }
              }
            }
            """
            
            response = requests.post(
                self.api_url,
                json={"query": query},
                headers=self.headers
            )
            
            self.logger.info(f"Linear API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and "teams" in data["data"]:
                    teams = []
                    for team_data in data["data"]["teams"]["nodes"]:
                        # Calculate members count from the nodes array
                        if "members" in team_data and "nodes" in team_data["members"]:
                            member_count = len(team_data["members"]["nodes"])
                            team_data["members"] = {"count": member_count}
                        teams.append(LinearTeam(team_data))
                    
                    self.logger.info(f"Retrieved {len(teams)} teams")
                    self.logger.info(f"Teams: {teams}")
                    return teams
                else:
                    self.logger.warning("No teams found")
                    return []
            else:
                self.logger.error(f"Error getting teams: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception getting teams: {str(e)}")
            return []
    
    def get_all_users(self) -> List[LinearUser]:
        """Get all users in the workspace."""
        self.logger.info("Getting all users in the workspace")
        
        try:
            query = """
            query Users {
              users {
                nodes {
                  id
                  name
                  email
                  displayName
                  active
                  admin
                  teams {
                    nodes {
                      id
                      name
                      key
                    }
                  }
                }
              }
            }
            """
            
            response = requests.post(
                self.api_url,
                json={"query": query},
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and "users" in data["data"]:
                    users = []
                    for user_data in data["data"]["users"]["nodes"]:
                        users.append(LinearUser(user_data))
                    
                    self.logger.info(f"Retrieved {len(users)} users")
                    return users
                else:
                    self.logger.warning("No users found")
                    return []
            else:
                self.logger.error(f"Error getting users: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception getting users: {str(e)}")
            return []
    
    def get_user_work_items(self, user_id: str, start_date: str, end_date: str) -> List[LinearWorkItem]:
        """
        Get work items for a specific user within a date range.
        
        Parameters:
        - user_id: The Linear user ID
        - start_date: ISO format date string (e.g., "2023-05-01")
        - end_date: ISO format date string (e.g., "2023-05-07")
        """
        self.logger.info(f"Getting work items for user {user_id} from {start_date} to {end_date}")
        
        try:
            query = """
            query UserWorkItems($userId: ID!, $startDate: DateTime!, $endDate: DateTime!) {
              user(id: $userId) {
                assignedIssues(
                  filter: {
                    updatedAt: { gte: $startDate, lte: $endDate }
                  }
                ) {
                  nodes {
                    id
                    title
                    description
                    state {
                      name
                    }
                    creator {
                      name
                    }
                    assignee {
                      name
                    }
                    createdAt
                    updatedAt
                    completedAt
                    cycle {
                      name
                    }
                    estimate
                    timeTracked
                  }
                }
              }
            }
            """
            
            variables = {
                "userId": user_id,
                "startDate": start_date,
                "endDate": end_date
            }
            
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and "user" in data["data"] and data["data"]["user"]:
                    work_items = []
                    for item_data in data["data"]["user"]["assignedIssues"]["nodes"]:
                        work_items.append(LinearWorkItem(item_data))
                    
                    self.logger.info(f"Retrieved {len(work_items)} work items for user {user_id}")
                    return work_items
                else:
                    self.logger.warning(f"User not found or no work items: {user_id}")
                    return []
            else:
                self.logger.error(f"Error getting user work items: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception getting user work items: {str(e)}")
            return []
    
    def get_user_time_entries(self, user_id: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get time entries for a specific user within a date range.
        
        Parameters:
        - user_id: The Linear user ID
        - start_date: ISO format date string (e.g., "2023-05-01")
        - end_date: ISO format date string (e.g., "2023-05-07")
        """
        self.logger.info(f"Getting time entries for user {user_id} from {start_date} to {end_date}")
        
        try:
            query = """
            query UserTimeEntries($userId: ID!, $startDate: DateTime!, $endDate: DateTime!) {
              user(id: $userId) {
                name
                timeEntries(
                  filter: {
                    createdAt: { gte: $startDate, lte: $endDate }
                  }
                ) {
                  nodes {
                    id
                    date
                    seconds
                    issue {
                      id
                      title
                    }
                  }
                }
              }
            }
            """
            
            variables = {
                "userId": user_id,
                "startDate": start_date,
                "endDate": end_date
            }
            
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and "user" in data["data"] and data["data"]["user"]:
                    user_data = data["data"]["user"]
                    time_entries = user_data.get("timeEntries", {}).get("nodes", [])
                    
                    self.logger.info(f"Retrieved {len(time_entries)} time entries for user {user_id}")
                    
                    # Calculate total time
                    total_seconds = sum(entry.get("seconds", 0) for entry in time_entries)
                    total_hours = round(total_seconds / 3600, 2)
                    
                    return {
                        "user_name": user_data.get("name", ""),
                        "total_hours": total_hours,
                        "total_seconds": total_seconds,
                        "entries": time_entries
                    }
                else:
                    self.logger.warning(f"User not found or no time entries: {user_id}")
                    return {
                        "user_name": "",
                        "total_hours": 0,
                        "total_seconds": 0,
                        "entries": []
                    }
            else:
                self.logger.error(f"Error getting user time entries: {response.text}")
                return {
                    "user_name": "",
                    "total_hours": 0,
                    "total_seconds": 0,
                    "entries": []
                }
                
        except Exception as e:
            self.logger.error(f"Exception getting user time entries: {str(e)}")
            return {
                "user_name": "",
                "total_hours": 0,
                "total_seconds": 0,
                "entries": []
            }
    
    def get_cycle_data(self, team_key: str = None, current: bool = True, cycle_number: int = None) -> Dict[str, Any]:
        """
        Get data about cycles for a team. Can retrieve the current cycle or a specific cycle by number.
        
        Parameters:
        - team_key: The Linear team key (e.g., "ENG"). If None, will try to get cycles across all teams.
        - current: If True, retrieves the current active cycle. If False, uses cycle_number.
        - cycle_number: The specific cycle number to retrieve. Only used if current=False.
        
        Returns:
        - A dictionary containing cycle data
        """
        self.logger.info(f"Getting cycle data for team: {team_key}, current: {current}, cycle_number: {cycle_number}")
        
        try:
            # First, we need to get the team ID from the team key
            if team_key:
                team_query = """
                query TeamByKey($teamKey: String!) {
                  teams(filter: {key: {eq: $teamKey}}) {
                    nodes {
                      id
                      name
                      key
                    }
                  }
                }
                """
                
                team_response = requests.post(
                    self.api_url,
                    json={"query": team_query, "variables": {"teamKey": team_key}},
                    headers=self.headers
                )
                
                if team_response.status_code != 200:
                    self.logger.error(f"Error getting team ID: {team_response.text}")
                    return {"error": f"Failed to retrieve team ID for key {team_key}"}
                
                team_data = team_response.json().get("data", {}).get("teams", {}).get("nodes", [])
                if not team_data:
                    return {"error": f"Team not found with key: {team_key}"}
                
                team_id = team_data[0]["id"]
                
                # Now build the query for cycle data with the team ID
                if current:
                    query = """
                    query TeamCurrentCycle($teamId: String!) {
                      team(id: $teamId) {
                        id
                        name
                        key
                        activeCycle {
                          id
                          name
                          number
                          startsAt
                          endsAt
                          completedAt
                          progress
                          issues {
                            nodes {
                              id
                              title
                              state {
                                name
                                type
                              }
                              assignee {
                                id
                                name
                              }
                              completedAt
                              estimate
                              priority
                            }
                          }
                        }
                      }
                    }
                    """
                    variables = {"teamId": team_id}
                else:
                    query = """
                    query TeamCycleByNumber($teamId: String!, $cycleNumber: Int!) {
                      team(id: $teamId) {
                        id
                        name
                        key
                        cycles(filter: {number: {eq: $cycleNumber}}) {
                          nodes {
                            id
                            name
                            number
                            startsAt
                            endsAt
                            completedAt
                            progress
                            issues {
                              nodes {
                                id
                                title
                                state {
                                  name
                                  type
                                }
                                assignee {
                                  id
                                  name
                                }
                                completedAt
                                estimate
                                priority
                              }
                            }
                          }
                        }
                      }
                    }
                    """
                    variables = {"teamId": team_id, "cycleNumber": cycle_number}
            else:
                # Get all active cycles across teams
                query = """
                query AllTeamsCycles {
                  teams {
                    nodes {
                      id
                      name
                      key
                      activeCycle {
                        id
                        name
                        number
                        startsAt
                        endsAt
                        completedAt
                        progress
                      }
                      cycles(first: 5) {
                        nodes {
                          id
                          name
                          number
                          startsAt
                          endsAt
                          completedAt
                          progress
                        }
                      }
                    }
                  }
                }
                """
                variables = {}
            
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error getting cycle data: {response.text}")
                return {"error": f"Failed to retrieve cycle data: {response.status_code}"}
            
            data = response.json().get("data", {})
            
            # Process the response based on the query type
            if team_key:
                team_data = data.get("team", {})
                if not team_data:
                    return {"error": f"Team not found: {team_key}"}
                
                cycle_data = team_data.get("activeCycle" if current else "cycle", {})
                if not cycle_data:
                    cycle_type = "current" if current else f"number {cycle_number}"
                    return {"error": f"No {cycle_type} cycle found for team {team_key}"}
                
                # Calculate additional metrics
                total_issues = len(cycle_data.get("issues", {}).get("nodes", []))
                completed_issues = sum(1 for issue in cycle_data.get("issues", {}).get("nodes", []) 
                                     if issue.get("completedAt"))
                
                return {
                    "team": {
                        "id": team_data.get("id"),
                        "name": team_data.get("name"),
                        "key": team_data.get("key")
                    },
                    "cycle": {
                        "id": cycle_data.get("id"),
                        "name": cycle_data.get("name"),
                        "number": cycle_data.get("number"),
                        "starts_at": cycle_data.get("startsAt"),
                        "ends_at": cycle_data.get("endsAt"),
                        "completed_at": cycle_data.get("completedAt"),
                        "progress": cycle_data.get("progress"),
                        "total_issues": total_issues,
                        "completed_issues": completed_issues,
                        "completion_percentage": round((completed_issues / total_issues) * 100, 2) if total_issues else 0,
                        "issues": cycle_data.get("issues", {}).get("nodes", [])
                    }
                }
            else:
                # Process data for all teams
                teams_data = data.get("teams", {}).get("nodes", [])
                teams_with_cycles = []
                
                for team in teams_data:
                    active_cycle = team.get("activeCycle", {})
                    recent_cycles = team.get("cycles", {}).get("nodes", [])
                    
                    teams_with_cycles.append({
                        "team": {
                            "id": team.get("id"),
                            "name": team.get("name"),
                            "key": team.get("key")
                        },
                        "active_cycle": active_cycle if active_cycle else None,
                        "recent_cycles": recent_cycles
                    })
                
                return {
                    "teams_count": len(teams_with_cycles),
                    "teams": teams_with_cycles
                }
                
        except Exception as e:
            self.logger.error(f"Exception getting cycle data: {str(e)}")
            return {"error": str(e)}
    
    def get_comprehensive_cycle_data(self, cycle_id: str) -> Dict[str, Any]:
        """Get comprehensive data for a specific cycle."""
        self.logger.info(f"Getting comprehensive data for cycle: {cycle_id}")
        
        try:
            # Step 1: Get basic cycle details with more detailed error logging
            cycle_query = """
            query CycleBasicDetails($id: String!) {
              cycle(id: $id) {
                id
                name
                number
                startsAt
                endsAt
                completedAt
                progress
                team {
                  id
                  name
                  key
                }
              }
            }
            """
            
            variables = {"id": cycle_id}
            
            # Add debugging to track the exact response
            self.logger.info(f"Querying cycle basic details for cycle ID: {cycle_id}")
            response = requests.post(
                self.api_url,
                json={"query": cycle_query, "variables": variables},
                headers=self.headers
            )
            
            self.logger.info(f"Cycle basic details response status: {response.status_code}")
            
            if response.status_code != 200:
                self.logger.error(f"Error getting basic cycle data: {response.text}")
                return {"error": f"Failed to retrieve cycle data: {response.status_code}"}
            
            response_data = response.json()
            
            # Debug the actual response structure
            self.logger.debug(f"Cycle query response: {json.dumps(response_data)[:500]}...")
            
            if "errors" in response_data:
                error_msg = response_data["errors"][0]["message"] if response_data["errors"] else "Unknown GraphQL error"
                self.logger.error(f"GraphQL error: {error_msg}")
                return {"error": error_msg}
            
            # Check for data at each level to identify where NoneType might be occurring
            if "data" not in response_data:
                self.logger.error("No 'data' field in cycle query response")
                return {"error": "Missing data field in API response"}
            
            data = response_data.get("data", {})
            if "cycle" not in data:
                self.logger.error("No 'cycle' field in cycle query data")
                return {"error": "Missing cycle field in API response"}
            
            cycle_data = data.get("cycle", {})
            if not cycle_data:
                self.logger.warning(f"Cycle not found: {cycle_id}")
                return {"error": f"Cycle not found: {cycle_id}"}
            
            # Create a safe structure for comprehensive data with defaults
            comprehensive_data = {
                "cycle": {
                    "id": cycle_data.get("id", ""),
                    "name": cycle_data.get("name", ""),
                    "number": cycle_data.get("number", 0),
                    "starts_at": cycle_data.get("startsAt", ""),
                    "ends_at": cycle_data.get("endsAt", ""),
                    "completed_at": cycle_data.get("completedAt"),
                    "progress": cycle_data.get("progress", 0),
                    "team": {}
                },
                "teams": [],
                "issues": [],
                "summary": {
                    "total_issues": 0,
                    "total_teams": 0,
                    "cycle_duration_days": 0,
                    "overall_completion_rate": 0,
                    "completed_issues": 0
                }
            }
            
            # Extract cycle date range with safe access
            start_date = cycle_data.get("startsAt", "")
            end_date = cycle_data.get("endsAt", "")
            
            # Safely extract team data
            if cycle_data.get("team"):
                team_data = cycle_data.get("team", {})
                comprehensive_data["cycle"]["team"] = {
                    "id": team_data.get("id", ""),
                    "name": team_data.get("name", ""),
                    "key": team_data.get("key", "")
                }
            
            # Step 2: Get issues with safer error handling
            issues = []
            try:
                self.logger.info(f"Fetching issues for cycle {cycle_id}")
                has_more_issues = True
                after_cursor = None
                page_size = 50
                
                while has_more_issues:
                    # Use a simplified query if needed
                    issues_query = """
                    query CycleIssues($id: String!, $first: Int!, $after: String) {
                      cycle(id: $id) {
                        issues(first: $first, after: $after) {
                          nodes {
                            id
                            number
                            title
                            description
                            state {
                              name
                              type
                            }
                            parent {
                              id
                              title
                            }
                            children {
                              nodes {
                                id
                                title
                              }
                            }
                            assignee {
                              id
                              name
                            }
                            team {
                              id
                              name
                              key
                            }
                            priority
                            createdAt
                            updatedAt
                            completedAt
                            estimate
                            comments {
                              nodes {
                                id
                                body
                                user {
                                  name
                                }
                                createdAt
                              }
                            }
                          }
                          pageInfo {
                            hasNextPage
                            endCursor
                          }
                        }
                      }
                    }
                    """
                    
                    issues_variables = {
                        "id": cycle_id,
                        "first": page_size,
                        "after": after_cursor
                    }
                    
                    issues_response = requests.post(
                        self.api_url,
                        json={"query": issues_query, "variables": issues_variables},
                        headers=self.headers
                    )
                    
                    if issues_response.status_code != 200:
                        self.logger.error(f"Error fetching cycle issues: {issues_response.text}")
                        break
                    
                    issues_response_data = issues_response.json()
                    
                    # Debug the response structure
                    self.logger.debug(f"Issues query response: {json.dumps(issues_response_data)[:500]}...")
                    
                    if "errors" in issues_response_data:
                        error_msg = issues_response_data["errors"][0]["message"] if issues_response_data["errors"] else "Unknown GraphQL error"
                        self.logger.error(f"GraphQL error in issues query: {error_msg}")
                        break
                    
                    issues_data = issues_response_data.get("data", {})
                    cycle_issues = issues_data.get("cycle", {})
                    
                    if not cycle_issues:
                        self.logger.warning("No cycle data returned in issues query")
                        break
                        
                    issues_nodes = cycle_issues.get("issues", {}).get("nodes", [])
                    self.logger.info(f"Retrieved {len(issues_nodes)} issues in this page")
                    issues.extend(issues_nodes)
                    
                    page_info = cycle_issues.get("issues", {}).get("pageInfo", {})
                    has_more_issues = page_info.get("hasNextPage", False)
                    after_cursor = page_info.get("endCursor") if has_more_issues else None
                
                self.logger.info(f"Total issues retrieved for cycle {cycle_id}: {len(issues)}")
            except Exception as e:
                self.logger.error(f"Error while fetching issues: {str(e)}")
                # Continue with any issues we've collected so far
            
            # Step 3: Get teams data with better error handling
            teams_info = []
            try:
                teams_query = """
                query TeamsBasic {
                  teams(first: 20) {
                    nodes {
                      id
                      name
                      key
                    }
                  }
                }
                """
                
                teams_response = requests.post(
                    self.api_url,
                    json={"query": teams_query},
                    headers=self.headers
                )
                
                if teams_response.status_code == 200:
                    teams_response_data = teams_response.json()
                    if "errors" not in teams_response_data:
                        teams_data = teams_response_data.get("data", {}).get("teams", {}).get("nodes", [])
                        teams_info = [{"id": team.get("id"), "name": team.get("name"), "key": team.get("key")} 
                                    for team in teams_data]
            except Exception as e:
                self.logger.error(f"Error fetching teams: {str(e)}")
            
            # Update the comprehensive data with what we've collected
            comprehensive_data["issues"] = [self._issue_to_dict(issue) for issue in issues]
            comprehensive_data["teams"] = teams_info
            comprehensive_data["summary"] = {
                "total_issues": len(issues),
                "total_teams": len(teams_info),
                "cycle_duration_days": self._calculate_date_diff(start_date, end_date),
                "overall_completion_rate": cycle_data.get("progress", 0),
                "completed_issues": sum(1 for issue in issues if issue.get("completedAt"))
            }
            
            self.logger.info(f"Successfully retrieved comprehensive data for cycle {cycle_id}")
            return comprehensive_data
            
        except Exception as e:
            self.logger.error(f"Exception getting comprehensive cycle data: {str(e)}, traceback: {traceback.format_exc()}")
            return {"error": str(e)}

    def _work_item_to_dict(self, work_item: LinearWorkItem) -> Dict[str, Any]:
        """Convert a LinearWorkItem to a dictionary."""
        return {
            "id": work_item.id,
            "title": work_item.title,
            "description": work_item.description,
            "state": work_item.state,
            "creator": work_item.creator,
            "assignee": work_item.assignee,
            "created_at": work_item.created_at,
            "updated_at": work_item.updated_at,
            "completed_at": work_item.completed_at,
            "cycle": work_item.cycle,
            "estimate": work_item.estimate,
            "time_spent": work_item.time_spent,
            "time_logged": work_item.time_logged
        }

    def _issue_to_dict(self, issue: Dict[str, Any]) -> Dict[str, Any]:
        """Convert an issue data dictionary to a standardized format."""
        # Process comments with better error handling
        comments = []
        if issue.get("comments") and issue.get("comments", {}).get("nodes"):
            for comment in issue.get("comments", {}).get("nodes", []):
                # Skip invalid comments
                if not comment:
                    continue
                    
                # Safely extract user name
                user_name = "Unknown User"
                if comment.get("user"):
                    user_name = comment.get("user", {}).get("name", "Unknown User")
                
                comments.append({
                    "body": comment.get("body", ""),
                    "user": user_name,
                    "created_at": comment.get("createdAt", "")
                })
        
        # Get team key and issue number
        team_key = issue.get("team", {}).get("key", "")
        issue_number = issue.get("number", 0)
        
        # Format ID as team_key-number if both are available
        issue_id = f"{team_key}-{issue_number}" if team_key and issue_number else issue.get("id", "")
        
        return {
            "id": issue_id,
            "number": issue_number,  # Include the raw number for reference
            "title": issue.get("title"),
            "description": issue.get("description"),
            "state": issue.get("state", {}).get("name") if issue.get("state") else "",
            "state_type": issue.get("state", {}).get("type") if issue.get("state") else "",
            "parent": {
                "id": issue.get("parent", {}).get("id") if issue.get("parent") else "",
                "title": issue.get("parent", {}).get("title") if issue.get("parent") else ""
            },
            "children": [{"id": child.get("id"), "title": child.get("title")} 
                       for child in issue.get("children", {}).get("nodes", [])] if issue.get("children") else [],
            "assignee": {
                "id": issue.get("assignee", {}).get("id") if issue.get("assignee") else "",
                "name": issue.get("assignee", {}).get("name") if issue.get("assignee") else ""
            },
            "team": {
                "id": issue.get("team", {}).get("id") if issue.get("team") else "",
                "name": issue.get("team", {}).get("name") if issue.get("team") else "",
                "key": team_key
            },
            "comments": comments,  # Use our safely processed comments
            "labels": [{"name": label.get("name"), "color": label.get("color")} 
                      for label in issue.get("labels", {}).get("nodes", [])],
            "priority": issue.get("priority"),
            "created_at": issue.get("createdAt"),
            "updated_at": issue.get("updatedAt"),
            "completed_at": issue.get("completedAt"),
            "estimate": issue.get("estimate"),
            "project": issue.get("project", {}).get("name") if issue.get("project") else "",
            "comments_count": len(issue.get("comments", {}).get("nodes", [])) if issue.get("comments") else 0,
            "history_events": len(issue.get("history", {}).get("nodes", [])) if issue.get("history") else 0
        }

    def _calculate_date_diff(self, start_date: str, end_date: str) -> int:
        """Calculate the difference in days between two ISO format dates."""
        if not start_date or not end_date:
            return 0
        
        try:
            start = datetime.datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end = datetime.datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            return (end - start).days
        except Exception:
            return 0

    def _calculate_overall_completion(self, cycle_data: Dict[str, Any]) -> float:
        """Calculate the overall completion rate for a cycle."""
        completed_scope = cycle_data.get("completedScope", 0)
        total_scope = cycle_data.get("scope", 0)
        
        if not total_scope:
            return 0
        
        return round((completed_scope / total_scope) * 100, 2)

    def get_team_recent_cycles(self, team_key: str, limit: int = 10) -> Dict[str, Any]:
        """
        Get recent cycles for a specific team, including the active cycle and completed cycles.
        
        Parameters:
        - team_key: The Linear team key (e.g., "ENG")
        - limit: Maximum number of recent cycles to retrieve
        
        Returns:
        - A dictionary containing team info and cycles data
        """
        self.logger.info(f"Getting recent cycles for team: {team_key}, limit: {limit}")
        
        try:
            # First, get the team ID from the team key
            team_query = """
            query TeamByKey($teamKey: String!) {
              teams(filter: {key: {eq: $teamKey}}) {
                nodes {
                  id
                  name
                  key
                }
              }
            }
            """
            
            team_response = requests.post(
                self.api_url,
                json={"query": team_query, "variables": {"teamKey": team_key}},
                headers=self.headers
            )
            
            if team_response.status_code != 200:
                self.logger.error(f"Error getting team ID: {team_response.text}")
                return {"error": f"Failed to retrieve team ID for key {team_key}"}
            
            team_data = team_response.json().get("data", {}).get("teams", {}).get("nodes", [])
            if not team_data:
                return {"error": f"Team not found with key: {team_key}"}
            
            team_id = team_data[0]["id"]
            
            # Build query to get both active cycle and recent cycles with issue information
            query = """
            query TeamCycles($teamId: String!, $first: Int!) {
              team(id: $teamId) {
                id
                name
                key
                activeCycle {
                  id
                  name
                  number
                  startsAt
                  endsAt
                  completedAt
                  progress
                  issues {
                    nodes {
                      id
                    }
                  }
                }
                # Get recent cycles 
                cycles(first: $first) {
                  nodes {
                    id
                    name
                    number
                    startsAt
                    endsAt
                    completedAt
                    progress
                    issues {
                      nodes {
                        id
                      }
                    }
                  }
                }
              }
            }
            """
            
            variables = {
                "teamId": team_id,
                "first": 20  # Request more cycles than needed
            }
            
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            
            if response.status_code != 200:
                self.logger.error(f"Error getting team cycles: {response.text}")
                return {"error": f"Failed to retrieve cycles data: {response.status_code}"}
            
            data = response.json().get("data", {})
            
            team_data = data.get("team", {})
            if not team_data:
                return {"error": f"Team not found: {team_key}"}
            
            active_cycle = team_data.get("activeCycle", {})
            all_cycles = team_data.get("cycles", {}).get("nodes", [])
            
            # Calculate issue counts for each cycle
            if active_cycle:
                issue_nodes = active_cycle.get("issues", {}).get("nodes", [])
                active_cycle["issue_count"] = len(issue_nodes)
            
            for cycle in all_cycles:
                issue_nodes = cycle.get("issues", {}).get("nodes", [])
                cycle["issue_count"] = len(issue_nodes)
            
            # Sort cycles by number (highest first)
            all_cycles.sort(key=lambda c: c.get("number", 0), reverse=True)
            
            # Filter out the active cycle from all cycles to avoid duplication
            if active_cycle and active_cycle.get("id"):
                active_id = active_cycle.get("id")
                all_cycles = [cycle for cycle in all_cycles if cycle.get("id") != active_id]
            
            # Only keep cycles that have issues
            # Ignore future cycles
            current_date = datetime.now().isoformat()
            valid_cycles = []
            
            for cycle in all_cycles:
                # Check if cycle has issues
                issue_count = cycle.get("issue_count", 0)
                
                # Check if cycle start date is not in the future
                has_start_date = bool(cycle.get("startsAt", ""))
                start_not_in_future = not has_start_date or cycle.get("startsAt", "") <= current_date
                
                if issue_count > 0 and start_not_in_future:
                    valid_cycles.append(cycle)
            
            # Add the active cycle if it has issues
            if active_cycle and active_cycle.get("issue_count", 0) > 0:
                valid_cycles.insert(0, active_cycle)
                
            # Limit to the requested number of cycles
            recent_cycles = valid_cycles[:limit] if len(valid_cycles) > limit else valid_cycles
            
            self.logger.info(f"Retrieved {len(recent_cycles)} valid cycles for team {team_key}")
            self.logger.info(f"Cycle numbers: {[c.get('number', 0) for c in recent_cycles]}")
            
            # Process and return the data
            return {
                "team": {
                    "id": team_data.get("id"),
                    "name": team_data.get("name"),
                    "key": team_data.get("key")
                },
                "active_cycle": active_cycle if active_cycle else None,
                "recent_cycles": recent_cycles[1:] if active_cycle else recent_cycles,  # Remove active cycle from recent cycles
                "cycles_count": len(recent_cycles) - (1 if active_cycle else 0)
            }
                
        except Exception as e:
            self.logger.error(f"Exception getting team recent cycles: {str(e)}")
            return {"error": str(e)}
    
    def get_multiple_cycles_data(self, cycles_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Get comprehensive data for multiple cycles efficiently.
        
        Parameters:
        - cycles_info: A list of dictionaries containing cycle information, each with at least 'id' key
        
        Returns:
        - A list of comprehensive cycle data dictionaries
        """
        self.logger.info(f"Getting comprehensive data for {len(cycles_info)} cycles")
        
        results = []
        
        for cycle_info in cycles_info:
            cycle_id = cycle_info.get('id')
            if not cycle_id:
                self.logger.warning(f"Skipping cycle with no ID: {cycle_info}")
                continue
                
            # Get comprehensive data for this cycle
            cycle_data = self.get_comprehensive_cycle_data(cycle_id)
            
            # If successful, add to results with team context
            if "error" not in cycle_data:
                # Add the team key from the original info if available
                if "team_key" in cycle_info and "team" in cycle_data["cycle"]:
                    cycle_data["cycle"]["team"]["key"] = cycle_info.get("team_key")
                
                results.append(cycle_data)
            else:
                self.logger.error(f"Error getting data for cycle {cycle_id}: {cycle_data.get('error')}")
        
        self.logger.info(f"Successfully retrieved data for {len(results)} out of {len(cycles_info)} cycles")
        return results
        

    async def create_linear_issue(self, title: str, description: str, assignee_name: str = None, team_key: str = None, priority: int = None) -> Dict[str, Any]:
        """
        Create a new Linear issue with the specified parameters.
        """
        try:
            # First, get the team ID if team_key is provided
            team_id = None
            if team_key:
                team_query = """
                query TeamByKey($teamKey: String!) {
                  teams(filter: {key: {eq: $teamKey}}) {
                    nodes {
                      id
                      states {
                        nodes {
                          id
                          name
                          type
                        }
                      }
                    }
                  }
                }
                """
                team_response = requests.post(
                    self.api_url,
                    json={"query": team_query, "variables": {"teamKey": team_key}},
                    headers=self.headers
                )
                
                team_data = team_response.json().get("data", {}).get("teams", {}).get("nodes", [])
                if team_data:
                    team_id = team_data[0]["id"]
                    # Get the "Todo" state ID from the team's states
                    states = team_data[0].get("states", {}).get("nodes", [])
                    todo_state = next((state for state in states if state["name"] == "Todo"), None)
                    if todo_state:
                        state_id = todo_state["id"]
                    else:
                        return {"error": "Could not find 'Todo' state for the team"}
                else:
                    return {"error": f"Team with key '{team_key}' not found"}
            else:
                return {"error": "team_key is required to create an issue"}

            # Get assignee ID if provided
            assignee_id = None
            if assignee_name:
                assignee_name = assignee_name.lstrip('@')
                user_query = """
                query UserByName($name: String!) {
                  users(filter: {name: {eq: $name}}) {
                    nodes {
                      id
                    }
                  }
                }
                """
                user_response = requests.post(
                    self.api_url,
                    json={"query": user_query, "variables": {"name": assignee_name}},
                    headers=self.headers
                )
                
                user_data = user_response.json().get("data", {}).get("users", {}).get("nodes", [])
                if user_data:
                    assignee_id = user_data[0]["id"]
                else:
                    return {"error": f"User '{assignee_name}' not found"}

            # Create the mutation query
            mutation = """
            mutation CreateIssue($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success
                issue {
                  id
                  identifier
                  title
                  url
                }
              }
            }
            """

            # Prepare input variables
            input_vars = {
                "title": title,
                "description": description,
                "teamId": team_id,
                "priority": priority if priority is not None else 0,
                "stateId": state_id  # Use stateId instead of state object
            }
            
            # Only add assigneeId if we have one
            if assignee_id:
                input_vars["assigneeId"] = assignee_id

            variables = {
                "input": input_vars
            }

            # Create the issue
            response = requests.post(
                self.api_url,
                json={"query": mutation, "variables": variables},
                headers=self.headers
            )

            if response.status_code != 200:
                return {"error": f"API request failed: {response.text}"}

            data = response.json()
            
            if "errors" in data:
                return {"error": data["errors"][0]["message"]}

            result = data.get("data", {}).get("issueCreate", {})
            if result.get("success"):
                issue = result.get("issue", {})
                return {
                    "success": True,
                    "issue": {
                        "id": issue.get("id"),
                        "identifier": issue.get("identifier"),
                        "title": issue.get("title"),
                        "url": issue.get("url"),
                        "assignee": assignee_name if assignee_name else None,
                        "team": team_key if team_key else None,
                        "priority": priority
                    }
                }
            else:
                return {"error": "Failed to create issue"}

        except Exception as e:
            self.logger.error(f"Error creating Linear issue: {str(e)}")
            return {"error": f"Error creating issue: {str(e)}"}
    
if __name__ == "__main__":
    async def run_tests():
        client = LinearClient(api_key=os.getenv('LINEAR_API_KEY'))

        # Test getting all users
        print("\nFetching all Linear users:")
        print("=" * 50)
        users = client.get_all_users()
        if users:
            print(f"Found {len(users)} users:")
            for user in users:
                print("\nUser Details:")
                print(f"Name: {user.name}")
                print(f"Display Name: {user.display_name}")
                print(f"Email: {user.email}")
                print(f"Active: {'Yes' if user.active else 'No'}")
                print(f"Teams: {', '.join([team.get('key', 'Unknown') for team in user.teams])}")
                print("-" * 30)
        else:
            print("❌ No users found or error occurred")
        
        print("\n" + "=" * 50)

        # Test issue creation
        print("\nTesting issue creation:")
        print("=" * 50)
        test_issue = await client.create_linear_issue(
            title="Test Issue from API Client",
            description="This is a test issue created via the Linear API client",
            assignee_name="Phát -",
            team_key="OPS",
            priority=2
        )
        
        if test_issue.get("success"):
            issue = test_issue["issue"]
            print(f"✅ Issue created successfully!")
            print(f"Identifier: {issue['identifier']}")
            print(f"Title: {issue['title']}")
            print(f"URL: {issue['url']}")
            print(f"Assigned to: {issue['assignee']}")
            print(f"Team: {issue['team']}")
            print(f"Priority: {issue['priority']}")
        else:
            print(f"❌ Failed to create issue: {test_issue.get('error')}")
        
        print("=" * 50)
    # Run the async function
    asyncio.run(run_tests())
    