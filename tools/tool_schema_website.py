SEARCH_WEBSITE_CONTENT = {
    "type": "function",
    "name": "search_website_content",
    "description": "Search the website for content based on a natural language query. Or by specific website type.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search the chunk content inside a RAG database storing website content."
            },
            "website_type": {
                "type": "string",
                "description": "The type of website to search. Main is tokenmetrics.com, Research is research.tokenmetrics.com, and Blog is newsletter.tokenmetrics.com.",
                "enum": ["main", "research", "blog"]
            },
            "limit": {
                "type": "integer",
                "description": "The maximum number of results to return. Default is 5."
            }
        },
        "required": [],
        "additionalProperties": False
    }
}

WEBSITE_SCHEMAS = {
    "search_website_content": SEARCH_WEBSITE_CONTENT
}