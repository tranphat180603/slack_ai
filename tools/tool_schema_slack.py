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
GET_USER_SCHEMA = {
    "type": "function",
    "name": "get_user",
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

# Schema for getting conversation context
GET_CONVERSATION_CONTEXT_SCHEMA = {
    "type": "function",
    "name": "get_conversation_context",
    "description": "Get the conversation context for the current chat session. This will automatically use the current channel and thread. Use this function when you need to summarize or refer to previous messages in the conversation.",
    "parameters": {
        "type": "object",
        "properties": {
            "max_messages": {
                "type": "integer",
                "description": "Maximum number of messages to retrieve from the conversation history. Default is 10. Use a larger number (e.g., 20-30) for longer context, or a smaller number (e.g., 5) for just recent messages."
            }
        },
        "additionalProperties": False
    }
}



# Schema for workspace message search
SEARCH_WORKSPACE_MESSAGES_SCHEMA = {
    "type": "function",
    "name": "search_workspace_messages",
    "description": "Search messages across entire Slack workspace with advanced filtering. Use this for finding specific content, discussions, or mentions across all channels you have access to.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string. Can include keywords, phrases, or Slack search operators."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (1-100). Default is 50."
            },
            "channels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of channel IDs to limit search to specific channels."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# Schema for file search
SEARCH_FILES_SCHEMA = {
    "type": "function",
    "name": "search_files",
    "description": "Search for files across the entire Slack workspace. Find documents, images, and other attachments by name or content.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for file names or content."
            },
            "file_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional filter by file types (e.g., ['pdf', 'doc', 'png'])."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of files to return (1-100). Default is 20."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

# Schema for getting workspace channels
GET_WORKSPACE_CHANNELS_SCHEMA = {
    "type": "function",
    "name": "get_workspace_channels",
    "description": "Get list of all channels in the workspace with metadata. Useful for discovering channels and understanding workspace structure.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_private": {
                "type": "boolean",
                "description": "Whether to include private channels (if you have access). Default is False."
            },
            "channel_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional filter by channel types."
            }
        },
        "additionalProperties": False
    }
}

# Schema for extracting action items
EXTRACT_ACTION_ITEMS_SCHEMA = {
    "type": "function",
    "name": "extract_action_items",
    "description": "Extract potential action items from recent channel conversations. Identifies messages that contain tasks, assignments, or follow-ups.",
    "parameters": {
        "type": "object",
        "properties": {
            "channel_id": {
                "type": "string",
                "description": "Channel ID to analyze for action items."
            },
            "days": {
                "type": "integer",
                "description": "Number of days to look back (1-30). Default is 7."
            }
        },
        "required": ["channel_id"],
        "additionalProperties": False
    }
}

# Schema for trending topics
GET_TRENDING_TOPICS_SCHEMA = {
    "type": "function",
    "name": "get_trending_topics",
    "description": "Identify trending discussion topics across the workspace based on message activity and channel engagement.",
    "parameters": {
        "type": "object",
        "properties": {
            "time_range": {
                "type": "string",
                "description": "Time range to analyze (e.g., '7d', '30d'). Default is '7d'."
            },
            "limit": {
                "type": "integer",
                "description": "Number of trending topics to return. Default is 10."
            }
        },
        "additionalProperties": False
    }
}

# Collection of all Slack schemas
SLACK_SCHEMAS = {
    "search_channel_history": SEARCH_CHANNEL_HISTORY_SCHEMA,
    "get_user": GET_USER_SCHEMA,
    "get_conversation_context": GET_CONVERSATION_CONTEXT_SCHEMA,
    "search_workspace_messages": SEARCH_WORKSPACE_MESSAGES_SCHEMA,
    "search_files": SEARCH_FILES_SCHEMA,
    "get_workspace_channels": GET_WORKSPACE_CHANNELS_SCHEMA,
    "extract_action_items": EXTRACT_ACTION_ITEMS_SCHEMA,
    "get_trending_topics": GET_TRENDING_TOPICS_SCHEMA
} 