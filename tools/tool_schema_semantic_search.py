"""
Schema definition for semantic search function.
This schema can be used with OpenAI function calling to find Linear content based on semantic similarity.
"""

# Schema for semantic search of Linear data
SEMANTIC_SEARCH_SCHEMA = {
    "type": "function",
    "name": "semantic_search_linear",
    "description": "IMPORTANT: Only use this for natural language concept searches. DO NOT use for finding specific issues by ID, number, or exact name - use filterIssues instead. This function performs semantic search to find content related to concepts expressed in natural language, across issues, projects, and comments in Linear. The search uses embeddings and semantic similarity, not exact matching. Leave parameters empty or omit them entirely rather than providing zero, empty string, or null values.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query describing concepts or topics (e.g., 'database migration problems' or 'UI responsiveness'). DO NOT use for specific issue lookups like 'Task #1234' or exact titles."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Omit rather than providing zero."
            },
            "use_reranker": {
                "type": "boolean",
                "description": "Whether to apply LLM-based reranking to improve search results. Only specify if needed."
            },
            "candidate_pool_size": {
                "type": "integer",
                "description": "Size of the initial candidate pool for reranking (only used when use_reranker=true). Omit rather than providing zero."
            },
            "team_key": {
                "type": "string",
                "description": "Filter results by team key. Omit rather than providing an empty string.",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "object_type": {
                "type": "string",
                "description": "Filter results by object type. Omit rather than providing an empty string.",
                "enum": ["Issue", "Project", "Comment"]
            }
        },
        "required": ["query"]
    }
}

# Collection of all semantic search schemas
SEMANTIC_SEARCH_SCHEMAS = {
    "semantic_search_linear": SEMANTIC_SEARCH_SCHEMA
}





