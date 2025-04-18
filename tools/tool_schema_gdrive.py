SEARCH_DRIVE_FILES_SCHEMA = {
    "type": "function",
    "name": "search_drive_files",
    "description": "Search Google Drive for files relevant to a natural language query. Returns metadata including file ID, name, mimeType, and webViewLink.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language query to search files by (e.g., 'salary bonus policies')."
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of file results to return.",
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}

GET_DRIVE_FILE_CONTENT_SCHEMA = {
    "type": "function",
    "name": "get_drive_file_content",
    "description": "Retrieve the full text content of a Google Drive file by its file ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_id": {
                "type": "string",
                "description": "The ID of the file to retrieve content for."
            }
        },
        "required": ["file_id"],
        "additionalProperties": False
    }
}

# Collection of all GDrive schemas
GDRIVE_SCHEMAS = {
    "search_drive_files": SEARCH_DRIVE_FILES_SCHEMA,
    "get_drive_file_content": GET_DRIVE_FILE_CONTENT_SCHEMA
}
