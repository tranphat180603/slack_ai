"""
Tools package for TMAI Agent.
This package contains schemas and implementations for interacting with external services.
"""

# Import schemas
from .tool_schema_linear import LINEAR_SCHEMAS
from .tool_schema_slack import SLACK_SCHEMAS
from .tool_schema_semantic_search import SEMANTIC_SEARCH_SCHEMAS
from .tool_schema_website import WEBSITE_SCHEMAS
from .tool_schema_gdrive import GDRIVE_SCHEMAS
# Import tools implementation
from .tools_declaration import (
    # Classes
    LinearTools,
    SlackTools,
    WebsiteTools,
    GDriveTools,
    # Exceptions
    LinearError,
    LinearAuthError,
    LinearNotFoundError,
    LinearValidationError,
    
    # Singleton instances
    linear_tools,
    slack_tools,
    website_tools,
    gdrive_tools,
)

# Export all schemas and tools
__all__ = [
    # Schema collections
    'LINEAR_SCHEMAS',
    'SLACK_SCHEMAS',
    'SEMANTIC_SEARCH_SCHEMAS',
    'WEBSITE_SCHEMAS',
    'GDRIVE_SCHEMAS',
    # Tool classes
    'LinearTools',
    'SlackTools',
    'WebsiteTools',
    'GDriveTools',
    # Exceptions
    'LinearError',
    'LinearAuthError',
    'LinearNotFoundError',
    'LinearValidationError',
    
    # Singleton instances
    'linear_tools',
    'slack_tools',
    'website_tools',
    'gdrive_tools',
] 