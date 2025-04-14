"""
Schema definitions for Slack API functions.
These schemas can be used with OpenAI function calling to interact with Slack.
"""

# Schema for searching channel history
SEARCH_CHANNEL_HISTORY_SCHEMA = {
    "type": "function",
    "name": "search_channel_history",
    "description": "Search Slack channel history with specific filters to find relevant messages. Results include content type detection (text, URLs, images, code blocks, etc.). Use this when you need to find messages from Slack channels. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "The ID of the Slack channel to search (e.g., 'C01234ABCDE')"
            },
            "username": {
                "type": "string",
                "description": "Filter messages by a specific username. You can use partial names or full names without the @ symbol. Example: 'john' will match 'John Smith'. Omit rather than providing an empty string."
            },
            "time_range": {
                "type": "string",
                "description": "Time unit to look back. Use 'hours' for recent messages (today), 'days' for messages within the last month, or 'weeks' for older messages. Omit if not needed.",
                "enum": ["hours", "days", "weeks"]
            },
            "time_value": {
                "type": "integer",
                "description": "Number of time units to look back. For example, 2 with time_range='days' means search the last 2 days. Valid range: 1-30. Omit rather than providing zero."
            },
            "message_count": {
                "type": "integer",
                "description": "Maximum number of messages to retrieve. Use smaller values (10-20) for focused searches, medium values (30-50) for general overviews, and larger values (50-100) for comprehensive searches. Omit rather than providing zero."
            }
        },
        "required": ["channel_id"],
        "additionalProperties": False
    }
}

# Schema for getting users
GET_USERS_SCHEMA = {
    "type": "function",
    "name": "get_users",
    "description": "Get user information for every employees in the company or a specific employee. By providing their display name, you can get the user's information. Or return all users if no display name is provided.",
    "parameters": {
        "type": "object",
        "properties": {
            "display_name": {
                "type": "string",
                "description": "The display name of the employee to search for (e.g. '@username'). You can search for the display names related to user's query. Omit rather than providing an empty string."
            }
        }
    },
    "additionalProperties": False
}

GET_CURRENT_USER_SCHEMA = {
    "type": "function",
    "name": "get_current_user",
    "description": "Get the current user's information on Slack, who are also the user. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {
                "type": "string",
                "description": "The ID of the user to get information for. Omit rather than providing an empty string."
            }
        }
    },
    "additionalProperties": False
}

# Schema for getting channels
GET_CHANNELS_SCHEMA = {
    "type": "function",
    "name": "get_channels",
    "description": "Get information about Slack channels for the workspace. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "ID of a specific channel to get information for. If you already have the channel ID, you can use it here. Omit rather than providing an empty string."
            },
            "channel_name": {
                "type": "string",
                "description": "Name of a channel to search for. You can provide a partial name to match. Omit rather than providing an empty string."
            }
        }
    },
    "additionalProperties": False
}

# Schema for getting channel members
GET_CHANNEL_MEMBERS_SCHEMA = {
    "type": "function",
    "name": "get_channel_members",
    "description": "Get members in a Slack channel. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "ID of the channel to get members for (e.g. 'C01234ABCDE')."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of members to retrieve. Omit rather than providing zero."
            }
        },
        "required": ["channel_id"],
        "additionalProperties": False
    }
}

# Schema for sending messages
SEND_MESSAGE_SCHEMA = {
    "type": "function",
    "name": "send_message",
    "description": "Send a message to a Slack channel or user. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "ID of the channel to send the message to (e.g. 'C01234ABCDE')."
            },
            "text": {
                "type": "string",
                "description": "The text of the message to send."
            }
        },
        "required": ["channel_id", "text"],
        "additionalProperties": False
    }
}

# Collection of all Slack schemas
SLACK_SCHEMAS = {
    "search_channel_history": SEARCH_CHANNEL_HISTORY_SCHEMA,
    "get_users": GET_USERS_SCHEMA,
    "get_current_user": GET_CURRENT_USER_SCHEMA,
    "get_channels": GET_CHANNELS_SCHEMA,
    "get_channel_members": GET_CHANNEL_MEMBERS_SCHEMA,
    "send_message": SEND_MESSAGE_SCHEMA
} 