"""
Schema definitions for Linear API functions.
These schemas can be used with OpenAI function calling to interact with Linear.
"""

# Schema for filtering issues
FILTER_ISSUES_SCHEMA = {
    "type": "function",
    "name": "filter_linear_issues",
    "description": "GraphQL-based function to filter Linear issues based on various criteria such as state, priority, assignee, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "The team key to filter issues by",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "state": {
                "type": "string",
                "description": "Filter by issue state (e.g. 'In Progress', 'Todo', 'Done')"
            },
            "priority": {
                "type": "number",
                "description": "Filter by priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)",
                "enum": [0.0, 1.0, 2.0, 3.0, 4.0]
            },
            "assignee_name": {
                "type": "string",
                "description": "Filter by assignee's display name. Supports exact match. Use assignee_contains for partial matches."
            },
            "assignee_contains": {
                "type": "string",
                "description": "Filter by assignee names containing this text (case-insensitive)"
            },
            "title_contains": {
                "type": "string",
                "description": "Filter issues where title contains this string"
            },
            "description_contains": {
                "type": "string",
                "description": "Filter issues where description contains this string"
            },
            "cycle_name": {
                "type": "string",
                "description": "Filter by cycle name"
            },
            "project_name": {
                "type": "string",
                "description": "Filter by project name"
            },
            "label_name": {
                "type": "string",
                "description": "Filter by label name"
            },
            "first": {
                "type": "number",
                "description": "Limit the number of issues returned"
            }
        },
        "required": ["team_key"],
        "additionalProperties": False
    }
}

# Schema for creating issues
CREATE_ISSUE_SCHEMA = {
    "type": "function",
    "name": "create_linear_issue",
    "description": "GraphQL-based function to create a new issue in Linear with specified details",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
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
                "description": "Markdown description of the issue"
            },
            "priority": {
                "type": "number",
                "description": "Priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)",
                "enum": [0.0, 1.0, 2.0, 3.0, 4.0]
            },
            "estimate": {
                "type": "number",
                "description": "Estimate points for the issue"
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
            "cycle_name": {
                "type": "string",
                "description": "Name of the cycle to add the issue to"
            },
            "parent_issue_number": {
                "type": "integer",
                "description": "Issue number of the parent issue"
            }
        },
        "required": ["team_key", "title"],
        "additionalProperties": False
    }
}

# Schema for updating issues
UPDATE_ISSUE_SCHEMA = {
    "type": "function",
    "name": "update_linear_issue",
    "description": "GraphQL-based function to update an existing issue in Linear",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {
                "type": "integer",
                "description": "The number of the issue to update"
            },
            "title": {
                "type": "string",
                "description": "New title for the issue"
            },
            "description": {
                "type": "string",
                "description": "New markdown description"
            },
            "priority": {
                "type": "number",
                "description": "New priority level (0.0: None, 1.0: Urgent, 2.0: High, 3.0: Medium, 4.0: Low)",
                "enum": [0.0, 1.0, 2.0, 3.0, 4.0]
            },
            "estimate": {
                "type": "number",
                "description": "New estimate points"
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
            "cycle_name": {
                "type": "string",
                "description": "Name of the cycle to move the issue to"
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
        "required": ["issue_number"],
        "additionalProperties": False
    }
}

# Schema for filtering comments
FILTER_COMMENTS_SCHEMA = {
    "type": "function",
    "name": "filter_linear_comments",
    "description": "GraphQL-based function to filter comments in Linear based on various criteria",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {
                "type": "integer",
                "description": "Filter comments by issue number"
            },
            "user_display_name": {
                "type": "string",
                "description": "Filter comments by user display name"
            },
            "contains": {
                "type": "string",
                "description": "Filter comments containing this text (case-sensitive)"
            },
            "contains_ignore_case": {
                "type": "string",
                "description": "Filter comments containing this text (case-insensitive)"
            }
        },
        "required": ["issue_number"],
        "additionalProperties": False
    }
}

# Schema for filtering attachments
FILTER_ATTACHMENTS_SCHEMA = {
    "type": "function",
    "name": "filter_linear_attachments",
    "description": "GraphQL-based function to filter attachments in Linear based on various criteria",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {
                "type": "integer",
                "description": "Filter attachments by issue number"
            },
            "title_contains": {
                "type": "string",
                "description": "Filter attachments where title contains this string"
            },
            "creator_display_name": {
                "type": "string",
                "description": "Filter attachments by creator display name"
            }
        },
        "required": ["issue_number"],
        "additionalProperties": False
    }
}

# Schema for getting all users in a team
GET_USERS_SCHEMA = {
    "type": "function",
    "name": "get_linear_team_users",
    "description": "GraphQL-based function to get all users in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "The team key to get users from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["team_key"],
        "additionalProperties": False
    }
}

# Schema for getting all projects in a team
GET_PROJECTS_SCHEMA = {
    "type": "function",
    "name": "get_linear_team_projects",
    "description": "GraphQL-based function to get all projects in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "The team key to get projects from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["team_key"],
        "additionalProperties": False
    }
}

# Schema for getting all cycles in a team
GET_CYCLES_SCHEMA = {
    "type": "function",
    "name": "get_linear_team_cycles",
    "description": "GraphQL-based function to get all cycles in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "The team key to get cycles from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["team_key"],
        "additionalProperties": False
    }
}

# Schema for getting all labels in a team
GET_LABELS_SCHEMA = {
    "type": "function",
    "name": "get_linear_team_labels",
    "description": "GraphQL-based function to get all labels in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "The team key to get labels from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["team_key"],
        "additionalProperties": False
    }
}

# Schema for getting all states in a team
GET_STATES_SCHEMA = {
    "type": "function",
    "name": "get_linear_team_states",
    "description": "GraphQL-based function to get all workflow states in a specified Linear team",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "The team key to get workflow states from",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            }
        },
        "required": ["team_key"],
        "additionalProperties": False
    }
}

# Schema for filtering projects
FILTER_PROJECTS_SCHEMA = {
    "type": "function",
    "name": "filter_linear_projects",
    "description": "GraphQL-based function to filter projects in Linear based on various criteria",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "Filter projects by team key",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "name": {
                "type": "string", 
                "description": "Exact match for project name"
            },
            "name_contains": {
                "type": "string",
                "description": "Filter projects where name contains this string (case-sensitive)"
            },
            "name_contains_ignore_case": {
                "type": "string",
                "description": "Filter projects where name contains this string (case-insensitive)"
            },
            "state": {
                "type": "string",
                "description": "Filter by project state (e.g. 'planned', 'started', 'completed')"
            },
            "lead_display_name": {
                "type": "string",
                "description": "Filter by project lead's display name"
            }
        }
    }
}

# Schema for filtering cycles
FILTER_CYCLES_SCHEMA = {
    "type": "function",
    "name": "filter_linear_cycles",
    "description": "GraphQL-based function to filter cycles in Linear based on various criteria",
    "parameters": {
        "type": "object",
        "properties": {
            "team_key": {
                "type": "string",
                "description": "Filter cycles by team key",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "name": {
                "type": "string",
                "description": "Exact match for cycle name"
            },
            "name_contains": {
                "type": "string",
                "description": "Filter cycles where name contains this string"
            },
            "is_active": {
                "type": "boolean",
                "description": "Filter for active/inactive cycles"
            }
        }
    }
}

# Schema for creating comments
CREATE_COMMENT_SCHEMA = {
    "type": "function",
    "name": "create_linear_comment",
    "description": "GraphQL-based function to create a new comment on a Linear issue",
    "parameters": {
        "type": "object",
        "properties": {
            "issue_number": {
                "type": "integer",
                "description": "The issue number to comment on"
            },
            "body": {
                "type": "string",
                "description": "The comment text (supports markdown)"
            }
        },
        "required": ["issue_number", "body"],
        "additionalProperties": False
    }
}

# Collection of all schemas
LINEAR_SCHEMAS = {
    "filter_issues": FILTER_ISSUES_SCHEMA,
    "create_issue": CREATE_ISSUE_SCHEMA,
    "update_issue": UPDATE_ISSUE_SCHEMA,
    "filter_comments": FILTER_COMMENTS_SCHEMA,
    "filter_attachments": FILTER_ATTACHMENTS_SCHEMA,
    "get_users": GET_USERS_SCHEMA,
    "get_projects": GET_PROJECTS_SCHEMA,
    "get_cycles": GET_CYCLES_SCHEMA,
    "get_labels": GET_LABELS_SCHEMA,
    "get_states": GET_STATES_SCHEMA,
    "filter_projects": FILTER_PROJECTS_SCHEMA,
    "filter_cycles": FILTER_CYCLES_SCHEMA,
    "create_comment": CREATE_COMMENT_SCHEMA
}


from typing import List
import json
from openai import OpenAI
import os
import dotenv

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
            tools=[LINEAR_SCHEMAS["filter_issues"]]
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
                                    if isinstance(parsed, dict) and 'team_key' in parsed:
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
            "team": {"key": {"eq": parameters["team_key"]}}
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
        
        if "cycle_name" in parameters:
            filter_criteria["cycle"] = {"name": {"eq": parameters["cycle_name"]}}
            
        if "project_name" in parameters:
            filter_criteria["project"] = {"name": {"eq": parameters["project_name"]}}
            
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
        
        # Import here to avoid circular imports
        from linear_db.linear_client import LinearClient
        
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