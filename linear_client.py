import requests
import os
from typing import Optional, Dict, Any, List
import logging
import datetime
from datetime import timedelta

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
        self.members_count = data.get("members", {}).get("totalCount", 0) if data.get("members") else 0

class LinearUser:
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", "")
        self.name = data.get("name", "")
        self.email = data.get("email", "")
        self.display_name = data.get("displayName", "")
        self.active = data.get("active", True)
        self.admin = data.get("admin", False)
        self.teams = [team for team in data.get("teams", {}).get("nodes", [])] if data.get("teams") else []

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

class LinearClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.linear.app/graphql"
        self.headers = {
            "Authorization": api_key,
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger("linear_client")
    
    def get_issue(self, issue_id: str) -> Optional[LinearIssue]:
        """Get a Linear issue by its ID."""
        self.logger.info(f"Getting Linear issue with ID: {issue_id}")
        
        try:
            query = """
            query Issue($id: String!) {
              issue(id: $id) {
                id
                title
                description
                state {
                  name
                }
                assignee {
                  name
                }
                labels {
                  nodes {
                    name
                  }
                }
                priority
                createdAt
              }
            }
            """
            
            variables = {"id": issue_id}
            
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            
            self.logger.info(f"Linear API response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.logger.info(f"Linear API response: {data}")
                
                if "data" in data and data["data"]["issue"]:
                    return LinearIssue(data["data"]["issue"])
                else:
                    self.logger.warning(f"Issue not found: {issue_id}")
                    return None
            else:
                self.logger.error(f"Error getting issue {issue_id}: {response.text}")
                return None
            
        except Exception as e:
            self.logger.error(f"Exception getting issue {issue_id}: {str(e)}")
            return None
    
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
                    totalCount
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
                        teams.append(LinearTeam(team_data))
                    
                    self.logger.info(f"Retrieved {len(teams)} teams")
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
    
    def get_team_members(self, team_id: str) -> List[LinearUser]:
        """Get all members of a specific team."""
        self.logger.info(f"Getting members for team ID: {team_id}")
        
        try:
            query = """
            query TeamMembers($teamId: ID!) {
              team(id: $teamId) {
                members {
                  nodes {
                    id
                    name
                    email
                    displayName
                    active
                    admin
                  }
                }
              }
            }
            """
            
            variables = {"teamId": team_id}
            
            response = requests.post(
                self.api_url,
                json={"query": query, "variables": variables},
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and "team" in data["data"] and data["data"]["team"]:
                    members = []
                    for member_data in data["data"]["team"]["members"]["nodes"]:
                        members.append(LinearUser(member_data))
                    
                    self.logger.info(f"Retrieved {len(members)} members for team {team_id}")
                    return members
                else:
                    self.logger.warning(f"Team not found: {team_id}")
                    return []
            else:
                self.logger.error(f"Error getting team members: {response.text}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception getting team members: {str(e)}")
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
    
    def get_user_time_entries(self, user_id: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
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
    
    def get_team_working_hours(self, team_id: str) -> Dict[str, Any]:
        """
        Get working hours for all members of a team for the current week.
        
        Parameters:
        - team_id: The Linear team ID
        """
        self.logger.info(f"Getting working hours for team {team_id} for the current week")
        
        try:
            # Get all team members
            members = self.get_team_members(team_id)
            
            if not members:
                self.logger.warning(f"No members found for team {team_id}")
                return {
                    "team_id": team_id,
                    "members": []
                }
            
            # Calculate the start and end of the current week (Monday to Sunday)
            today = datetime.datetime.now()
            start_of_week = today - timedelta(days=today.weekday())
            start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_week = start_of_week + timedelta(days=6, hours=23, minutes=59, seconds=59)
            
            # Format dates for the API
            start_date = start_of_week.isoformat() + "Z"
            end_date = end_of_week.isoformat() + "Z"
            
            # Get time entries for each member
            member_hours = []
            for member in members:
                time_data = self.get_user_time_entries(member.id, start_date, end_date)
                
                member_info = {
                    "id": member.id,
                    "name": member.name,
                    "email": member.email,
                    "display_name": member.display_name,
                    "total_hours": time_data.get("total_hours", 0),
                    "total_seconds": time_data.get("total_seconds", 0),
                    "required_hours": 40,  # Default, can be customized
                    "missing_hours": max(0, 40 - time_data.get("total_hours", 0)),  # Default 40 hour work week
                    "entries_count": len(time_data.get("entries", []))
                }
                
                member_hours.append(member_info)
            
            # Summarize team data
            team_data = {
                "team_id": team_id,
                "members": member_hours,
                "week_start_date": start_of_week.strftime("%Y-%m-%d"),
                "week_end_date": end_of_week.strftime("%Y-%m-%d"),
                "total_team_hours": sum(m.get("total_hours", 0) for m in member_hours),
                "avg_team_hours": round(sum(m.get("total_hours", 0) for m in member_hours) / len(member_hours), 2) if member_hours else 0
            }
            
            self.logger.info(f"Retrieved working hours for {len(member_hours)} members in team {team_id}")
            return team_data
                
        except Exception as e:
            self.logger.error(f"Exception getting team working hours: {str(e)}")
            return {
                "team_id": team_id,
                "members": [],
                "error": str(e)
            }
    
    def get_all_teams_working_hours(self) -> List[Dict[str, Any]]:
        """Get working hours for all teams in the workspace for the current week."""
        self.logger.info("Getting working hours for all teams")
        
        teams = self.get_all_teams()
        if not teams:
            self.logger.warning("No teams found in workspace")
            return []
        
        all_team_data = []
        for team in teams:
            team_hours = self.get_team_working_hours(team.id)
            all_team_data.append({
                "team_id": team.id,
                "team_name": team.name,
                "team_key": team.key,
                "week_data": team_hours
            })
        
        self.logger.info(f"Retrieved working hours for {len(all_team_data)} teams")
        return all_team_data
    
    def find_users_not_meeting_required_hours(self, required_hours: int = 40) -> List[Dict[str, Any]]:
        """
        Find all users across the workspace who haven't met their required working hours this week.
        
        Parameters:
        - required_hours: The number of required working hours per week (default: 40)
        """
        self.logger.info(f"Finding users not meeting required hours ({required_hours}h)")
        
        try:
            all_teams_data = self.get_all_teams_working_hours()
            
            users_not_meeting_hours = []
            
            for team_data in all_teams_data:
                team_week_data = team_data.get("week_data", {})
                team_members = team_week_data.get("members", [])
                
                for member in team_members:
                    member_hours = member.get("total_hours", 0)
                    
                    if member_hours < required_hours:
                        users_not_meeting_hours.append({
                            "user_id": member.get("id", ""),
                            "name": member.get("name", ""),
                            "email": member.get("email", ""),
                            "team_name": team_data.get("team_name", ""),
                            "team_key": team_data.get("team_key", ""),
                            "logged_hours": member_hours,
                            "required_hours": required_hours,
                            "missing_hours": required_hours - member_hours
                        })
            
            # Sort by missing hours (descending)
            users_not_meeting_hours.sort(key=lambda x: x.get("missing_hours", 0), reverse=True)
            
            self.logger.info(f"Found {len(users_not_meeting_hours)} users not meeting required hours")
            return users_not_meeting_hours
            
        except Exception as e:
            self.logger.error(f"Exception finding users not meeting hours: {str(e)}")
            return [] 