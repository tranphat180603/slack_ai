SEARCH_WEBSITE_CONTENT = {
    "type": "function",
    "name": "search_website_content",
    "description": "Search the website for content based on a natural language query. Or by specific website type. It only returns chunks of content, not the full content of the website.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search the chunk content inside a RAG database storing website content. This is derived from the plan to come up with the most appropriate answer to the user's request."
            },
            "website_type": {
                "type": "string",
                "description": "The type of website to search. Main is tokenmetrics.com, Research is research.tokenmetrics.com, and Blog is newsletter.tokenmetrics.com.",
                "enum": ["main", "research", "blog"]
            },
            "distinct_on_url": {
                "type": "boolean",
                "description": "Whether to search for distinct content across multiple sites of Token Metrics, by default it will prioritize the most relevant content  (or closest) to the query, but this flag set to True helps you search more general information about the website across many sites of Token Metrics. And maybe sometimes at the expense of relevance. So set this to False if you prioritize relevance over general information."
            },
            "return_full_content": {
                "type": "boolean",
                "description": "Whether to return the full content of the website which contains the chunk. This helps when you want more granular information about a specific website. So as a rule of thumb, only use this in combination with distinct_on_url set to True."
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