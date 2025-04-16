"""
Schema definitions for Linear API functions.
These schemas can be used with OpenAI function calling to interact with Linear.
"""

# Schema for filtering issues
FILTER_ISSUES_SCHEMA = {
    "type": "function",
    "name": "filterIssues",
    "description": "GraphQL-based function to filter Linear issues based on various criteria such as state, priority, assignee, etc. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "The team key to filter issues by",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "issue_number": {
                "type": "integer",
                "description": "Filter by specific issue number. Note: Issue numbers are only unique within a team, so it's recommended to use with teamKey."
            },
            "state": {
                "type": "string",
                "description": "Filter by issue state (e.g. 'In Progress', 'Todo', 'Done'). Omit if no specific state filter is needed."
            },
            "priority": {
                "type": "number",
                "description": "Filter by priority level (1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low). Omit this field rather than using 0 if no priority filter is needed.",
                "enum": [0.0, 1.0, 2.0, 3.0, 4.0]
            },
            "assignee_name": {
                "type": "string",
                "description": "Filter by assignee's display name. Supports exact match. Omit if no assignee filter is needed."
            },
            "assignee_contains": {
                "type": "string",
                "description": "Filter by assignee names containing this text (case-insensitive). Omit rather than providing an empty string."
            },
            "title_contains": {
                "type": "string",
                "description": "Filter issues where title contains this string. Omit rather than providing an empty string."
            },
            "description_contains": {
                "type": "string",
                "description": "Filter issues where description contains this string. Omit rather than providing an empty string."
            },
            "cycle_number": {
                "type": "integer",
                "description": "Filter by cycle number (numeric identifier). Omit rather than providing zero if no cycle filter is needed."
            },
            "project_id": {
                "type": "string",
                "description": "Filtesr by project ID. Omit rather than providing an empty string."
            },
            "label_name": {
                "type": "string",
                "description": "Filter by label name. Omit rather than providing an empty string."
            },
            "first": {
                "type": "number",
                "description": "Limit the number of issues returned. Use a positive integer. Determine a reasonable number of this as not to flush the context window."
            },
            "include_description": {
                "type": "boolean",
                "description": "Whether to include the description field in the results. Set to False most of the time. Just set the True if users request details about the issue."
            }
        },
        "required": ["teamKey"],
        "additionalProperties": False
    }
}

# Schema for creating issues
CREATE_ISSUE_SCHEMA = {
    "type": "function",
    "name": "createIssue",
    "description": "GraphQL-based function to create a new issue in Linear with specified details. The purpose of this function is to create context-aware, well-groomed issues in Linear.",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "The team key where the issue will be created",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "title": {
                "type": "string",
                "description": "Title of the issue"
            },
            "description": {
                "type": "string",
                "description": "Markdown description of the issue. Always try to groom the description carefully before creating the issue."
            },
            "priority": {
                "type": "number",
                "description": "Priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)",
                "enum": [0.0, 1.0, 2.0, 3.0, 4.0]
            },
            "estimate": {
                "type": "number",
                "description": "Estimate points for the issue",
                "enum": [1, 2, 3, 4, 5, 6, 7]
            },
            "assignee_name": {
                "type": "string",
                "description": "Display name of the user to assign the issue to"
            },
            "state_name": {
                "type": "string",
                "description": "Name of the workflow state (e.g. 'Todo', 'In Progress', 'Done')"
            },
            "label_names": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "List of label names to apply to the issue"
            },
            "project_name": {
                "type": "string",
                "description": "Name of the project to add the issue to"
            },
            "cycle_number": {
                "type": "integer",
                "description": "Number of the cycle to add the issue to"
            },
            "parent_issue_number": {
                "type": "integer",
                "description": "Issue number of the parent issue"
            }
        },
        "required": ["teamKey", "title"],
        "additionalProperties": False
    }
}

# Schema for updating issues
UPDATE_ISSUE_SCHEMA = {
    "type": "function",
    "name": "updateIssue",
    "description": "GraphQL-based function to update an existing issue in Linear. This function is to properly update the issue with the new details gathered Linear.",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "The team key to update the issue in (enum: ENG, OPS, RES, AI, MKT, PRO)"
            },
            "issue_number": {
                "type": "integer",
                "description": "The number of the issue to update"
            },
            "title": {
                "type": "string",
                "description": "Whatever you write here, will be the entire new title for the issue. If not changed, must repeat the old title. Do not leave it blank."
            },
            "description": {
                "type": "string",
                "description": "Whatever you write here, will be the entire new description for the issue. If not changed, must repeat the old description. Always try to groom the description carefully if the user requests."
            },
            "priority": {
                "type": "number",
                "description": "New priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)",
                "enum": [0.0, 1.0, 2.0, 3.0, 4.0]
            },
            "estimate": {
                "type": "number",
                "description": "New estimate points",
                "enum": [1, 2, 3, 4, 5, 6, 7]
            },
            "assignee_name": {
                "type": "string",
                "description": "Display name of the user to reassign to"
            },
            "state_name": {
                "type": "string",
                "description": "New workflow state name"
            },
            "label_names": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "New list of label names"
            },
            "project_name": {
                "type": "string",
                "description": "Name of the project to move the issue to"
            },
            "cycle_number": {
                "type": "integer",
                "description": "Number of the cycle to move the issue to"
            },
            "parent_issue_number": {
                "type": "integer",
                "description": "Issue number of the new parent issue"
            },
            "archived": {
                "type": "boolean",
                "description": "Whether to archive the issue"
            }
        },
        "required": ["team_key", "issue_number"],
        "additionalProperties": False
    }
}

# Schema for filtering comments
FILTER_COMMENTS_SCHEMA = {
    "type": "function",
    "name": "filterComments",
    "description": "GraphQL-based function to filter comments in Linear based on various criteria. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {
                "type": "integer",
                "description": "Filter comments by issue number"
            },
            "user_display_name": {
                "type": "string",
                "description": "Filter comments by user display name. Omit rather than providing an empty string."
            },
            "body_contains": {
                "type": "string",
                "description": "Filter comments where body contains this text. Omit rather than providing an empty string."
            }
        },
        "required": ["issue_number"],
        "additionalProperties": False
    }
}

# Schema for filtering attachments
FILTER_ATTACHMENTS_SCHEMA = {
    "type": "function",
    "name": "filterAttachments",
    "description": "GraphQL-based function to filter attachments in Linear based on various criteria. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {
                "type": "integer",
                "description": "Filter attachments by issue number"
            },
            "title_contains": {
                "type": "string",
                "description": "Filter attachments where title contains this string. Omit rather than providing an empty string."
            },
            "creator_display_name": {
                "type": "string",
                "description": "Filter attachments by creator display name. Omit rather than providing an empty string."
            }
        },
        "required": ["issue_number"],
        "additionalProperties": False
    }
}

# Schema for getting all users in a team
GET_USERS_SCHEMA = {
    "type": "function",
    "name": "getAllUsers",
    "description": "GraphQL-based function to get all users in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "The team key to get users from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["teamKey"],
        "additionalProperties": False
    }
}

GET_TEAMS_SCHEMA = {
    "type": "function",
    "name": "getAllTeams",
    "description": "GraphQL-based function to get all teams in Linear",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    }
}

# Schema for getting all projects in a team
GET_PROJECTS_SCHEMA = {
    "type": "function",
    "name": "getAllProjects",
    "description": "GraphQL-based function to get all projects in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "The team key to get projects from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["teamKey"],
        "additionalProperties": False
    }
}

# Schema for getting all cycles in a team
GET_CYCLES_SCHEMA = {
    "type": "function",
    "name": "getAllCycles",
    "description": "GraphQL-based function to get all cycles up to the current cycle in a specified Linear team. The largest cycle number is the current cycle and the rest are past cycles in chronological order.",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "The team key to get cycles from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["teamKey"],
        "additionalProperties": False
    }
}

# Schema for getting all labels in a team
GET_LABELS_SCHEMA = {
    "type": "function",
    "name": "getAllLabels",
    "description": "GraphQL-based function to get all labels in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "The team key to get labels from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["teamKey"],
        "additionalProperties": False
    }
}

# Schema for getting all states in a team
GET_STATES_SCHEMA = {
    "type": "function",
    "name": "getAllStates",
    "description": "GraphQL-based function to get all workflow states in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "The team key to get workflow states from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["teamKey"],
        "additionalProperties": False
    }
}

# Schema for filtering projects
FILTER_PROJECTS_SCHEMA = {
    "type": "function",
    "name": "filterProjects",
    "description": "GraphQL-based function to filter projects in Linear based on various criteria. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "Filter projects by team key. Only provide if you need to filter by team.",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "name": {
                "type": "string", 
                "description": "Exact match for project name. Omit rather than providing an empty string."
            },
            "name_contains": {
                "type": "string",
                "description": "Filter projects where name contains this string (case-sensitive). Omit rather than providing an empty string."
            },
            "state": {
                "type": "string",
                "description": "Filter by project state (e.g. 'planned', 'started', 'completed'). Omit rather than providing an empty string."
            },
            "lead_display_name": {
                "type": "string",
                "description": "Filter by project lead's display name. Omit rather than providing an empty string."
            }
        }
    }
}

# Schema for filtering cycles
FILTER_CYCLES_SCHEMA = {
    "type": "function",
    "name": "filterCycles",
    "description": "GraphQL-based function to filter cycles in Linear based on various criteria. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "teamKey": {
                "type": "string",
                "description": "Filter cycles by team key. Only provide if you need to filter by team.",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "number": {
                "type": "integer",
                "description": "Filter by cycle number (numeric identifier). Omit rather than providing zero if no specific cycle is needed."
            },
            "starts_at": {
                "type": "string",
                "description": "Filter by cycle start date (format: YYYY-MM-DD). Omit rather than providing an empty string."
            },
            "ends_at": {
                "type": "string",
                "description": "Filter by cycle end date (format: YYYY-MM-DD). Omit rather than providing an empty string."
            },
            "filter_by_start_date": {
                "type": "boolean",
                "description": "Whether to only include cycles that have started. Only specify if you need to change from the default (true)."
            }
        }
    }
}

# Schema for creating comments
CREATE_COMMENT_SCHEMA = {
    "type": "function",
    "name": "createComment",
    "description": "GraphQL-based function to create a new comment on a Linear issue",
    "parameters": {
        "type": "object",
        "properties": {
            "issueNumber": {
                "type": "integer",
                "description": "The issue number to comment on"
            },
            "teamKey": {
                "type": "string",
                "description": "The team key to comment on (enum: ENG, OPS, RES, AI, MKT, PRO)"
            },
            "commentData": {
                "type": "object",
                "description": "The comment data object containing the body text (supports markdown)",
                "properties": {
                    "body": {
                        "type": "string",
                        "description": "The comment text in markdown format"
                    }
                },
                "required": ["body"]
            }
        },
        "required": ["issueNumber", "teamKey", "commentData"],
        "additionalProperties": False
    }
}

# Schema for getting current user
GET_CURRENT_USER_SCHEMA = {
    "type": "function",
    "name": "getCurrentUser",
    "description": "Get information about a specific user by their Slack display name",
    "parameters": {
        "type": "object",
        "properties": {
            "slack_display_name": {
                "type": "string",
                "description": "The Slack display name of the user to look up (e.g. '@username')"
            }
        },
        "required": ["slack_display_name"],
        "additionalProperties": False
    }
}

# Schema for getting user message by number
GET_USER_MESSAGE_BY_NUMBER_SCHEMA = {
    "type": "function",
    "name": "getUserMessageByNumber",
    "description": "Retrieves the N most recent user messages from the current conversation thread. This is useful when you need to get the precise wording of recent messages, especially for long texts or when creating issues/tickets.",
    "parameters": {
        "type": "object",
        "properties": {
            "number": {
                "type": "integer",
                "description": "The number of most recent user messages to retrieve (e.g., 5 means retrieve the 5 most recent user messages)"
            }
        },
        "required": ["number"],
        "additionalProperties": False
    }
}

# Collection of all schemas
LINEAR_SCHEMAS = {
    "filterIssues": FILTER_ISSUES_SCHEMA,
    "createIssue": CREATE_ISSUE_SCHEMA,
    "updateIssue": UPDATE_ISSUE_SCHEMA,
    "filterComments": FILTER_COMMENTS_SCHEMA,
    "filterAttachments": FILTER_ATTACHMENTS_SCHEMA,
    "getAllUsers": GET_USERS_SCHEMA,
    "getAllTeams": GET_TEAMS_SCHEMA,
    "getAllProjects": GET_PROJECTS_SCHEMA,
    "getAllCycles": GET_CYCLES_SCHEMA,
    "getAllLabels": GET_LABELS_SCHEMA,
    "getAllStates": GET_STATES_SCHEMA,
    "filterProjects": FILTER_PROJECTS_SCHEMA,
    "filterCycles": FILTER_CYCLES_SCHEMA,
    "createComment": CREATE_COMMENT_SCHEMA,
    "getCurrentUser": GET_CURRENT_USER_SCHEMA,
    "getUserMessageByNumber": GET_USER_MESSAGE_BY_NUMBER_SCHEMA
}


from typing import List
import json
from openai import OpenAI
import os
import dotenv
from ops_linear_db.linear_client import LinearClient

dotenv.load_dotenv()

def test_linear_filter(user_query: str, openai_api_key: str) -> List[dict]:
    """
    Use OpenAI to interpret a natural language query and filter issues.
    
    Args:
        user_query: Natural language query about issues
        openai_api_key: OpenAI API key
        
    Returns:
        List of matching issues
        
    Raises:
        LinearError: For Linear API errors
        ValueError: For OpenAI API errors
    """
    if not openai_api_key:
        raise ValueError("OpenAI API key is required")
        
    if not user_query:
        raise ValueError("Query string is required")
        
    try:
        client = OpenAI(api_key=openai_api_key)
        
        # Add system guidance to the user message
        enhanced_query = "You are a helpful assistant designed to convert natural language queries about Linear issues into structured filters. Your job is to call the filter_linear_issues function with appropriate parameters. Here's the user's query: " + user_query
        
        response = client.responses.create(
            model="o3-mini",
            input=[
                {"role": "user", "content": enhanced_query}
            ],
            tools=[LINEAR_SCHEMAS["filterIssues"]]
        )
        
        # Print raw response
        print("\nRaw OpenAI Response:")
        print("--------------------")
        print(f"Model: {response.model}")
        print(f"Response type: {type(response)}")
        print(f"Response attributes: {dir(response)}")
        
        if hasattr(response, 'usage'):
            print(f"Input tokens: {response.usage.input_tokens}")
            print(f"Output tokens: {response.usage.output_tokens}")
            print(f"Total tokens: {response.usage.total_tokens}")
        
        # Print full response for debugging
        print("\nFull response:")
        print(response)
        
        if not hasattr(response, 'output'):
            raise ValueError("No output in OpenAI response")
            
        # Print output details for debugging
        print("\nOutput details:")
        for i, output_item in enumerate(response.output):
            print(f"Item {i}:")
            print(f"  Type: {getattr(output_item, 'type', 'unknown')}")
            for attr in dir(output_item):
                if not attr.startswith('_') and attr != 'type':
                    try:
                        value = getattr(output_item, attr)
                        print(f"  {attr}: {value}")
                    except:
                        print(f"  {attr}: <error getting value>")
                        
        # Find the function call in the output
        function_call = None
        for output_item in response.output:
            if hasattr(output_item, 'type') and output_item.type == 'function_call':
                function_call = output_item
                break
        
        # Get parameters either from function call or from message content
        if function_call:
            parameters = json.loads(function_call.arguments)
        else:
            # Check for message with content
            message_parameters = None
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == 'message':
                    if hasattr(output_item, 'content'):
                        for content_item in output_item.content:
                            print(f"Content item: {content_item}")
                            if hasattr(content_item, 'text'):
                                try:
                                    # Try to parse the text as JSON
                                    parsed = json.loads(content_item.text)
                                    if isinstance(parsed, dict) and 'teamKey' in parsed:
                                        print("Found parameters in text output!")
                                        message_parameters = parsed
                                        break
                                except:
                                    pass
            
            if message_parameters:
                parameters = message_parameters
            else:
                raise ValueError("No function call or valid parameters in OpenAI response")
        
        # Build the Linear API filter
        filter_criteria = {
            "team": {"key": {"eq": parameters["teamKey"]}}
        }
        
        if "state" in parameters:
            filter_criteria["state"] = {"name": {"eq": parameters["state"]}}
        
        if "priority" in parameters:
            try:
                priority = float(parameters["priority"])
                if priority not in [0.0, 1.0, 2.0, 3.0, 4.0]:
                    raise ValueError(f"Invalid priority value: {priority}")
                filter_criteria["priority"] = {"eq": priority}
            except ValueError as e:
                raise ValueError(f"Invalid priority format: {str(e)}")
        
        if "assignee_name" in parameters:
            filter_criteria["assignee"] = {"displayName": {"eq": parameters["assignee_name"]}}
            
        if "assignee_contains" in parameters:
            filter_criteria["assignee"] = {"displayName": {"containsIgnoreCase": parameters["assignee_contains"]}}
        
        if "title_contains" in parameters:
            filter_criteria["title"] = {"contains": parameters["title_contains"]}
            
        if "description_contains" in parameters:
            filter_criteria["description"] = {"contains": parameters["description_contains"]}
        
        if "cycle_number" in parameters:
            filter_criteria["cycle"] = {"number": {"eq": parameters["cycle_number"]}}
        
        if "project_id" in parameters:
            filter_criteria["project"] = {"id": {"eq": parameters["project_id"]}}
        
        if "label_name" in parameters:
            filter_criteria["labels"] = {"some": {"name": {"eq": parameters["label_name"]}}}
        
        # Extract limit if specified
        limit = parameters.get("first")
        if limit is not None:
            try:
                limit = int(limit)
                if limit < 1:
                    raise ValueError("Limit must be positive")
            except ValueError:
                raise ValueError(f"Invalid limit value: {limit}")
        
        print("\nGenerated Linear API Filter:")
        print(json.dumps(filter_criteria, indent=2))
        if limit:
            print(f"Limit: {limit}")
        print("--------------------\n")
        
        # Get the Linear client
        linear = LinearClient(os.getenv('LINEAR_API_KEY'))
        return linear.filterIssues(filter_criteria, limit=limit)
        
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON in OpenAI response")
    except Exception as e:
        if isinstance(e, ValueError):
            raise
        raise Exception(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    test_linear_filter("What are the issues in the 'Improve Lucky X Agent' project?", openai_api_key=os.getenv("OPENAI_API_KEY"))