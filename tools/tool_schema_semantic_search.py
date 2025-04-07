"""
Schema definition for semantic search function.
This schema can be used with OpenAI function calling to find Linear content based on semantic similarity.
"""

# Schema for semantic search of Linear data
SEMANTIC_SEARCH_SCHEMA = {
    "type": "function",
    "name": "semantic_search_linear",
    "description": "Find Linear content (by loosely matching the content of issues, projects, comments) semantically similar to a natural language query. Returns results with metadata that can be used for further filtering with other Linear functions. Used when the query mentions about a name or description of issues, projects, comments, etc.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search for (e.g., 'issues about improving the agent' or 'database migration problems')"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
            },
            "use_reranker": {
                "type": "boolean",
                "description": "Whether to apply LLM-based reranking to improve search results",
            },
            "candidate_pool_size": {
                "type": "integer",
                "description": "Size of the initial candidate pool for reranking (only used when use_reranker=true)",
            },
            "team_key": {
                "type": "string",
                "description": "Filter results by team key",
                "enum": ["ENG", "OPS", "RES", "AI", "MKT", "PRO"]
            },
            "object_type": {
                "type": "string",
                "description": "Filter results by object type",
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





