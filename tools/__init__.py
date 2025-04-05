"""
Tool Schemas for TMAI Agent.
This package contains JSONSchema definitions for tools that can be used with LLM function calling.
"""

from .tool_schema_semantic_search import SEMANTIC_SEARCH_SCHEMAS
from .tool_schema_linear import LINEAR_SCHEMAS
from .tool_schema_slack import SLACK_SCHEMAS

# Combine all schemas
ALL_SCHEMAS = {
    **SEMANTIC_SEARCH_SCHEMAS,
    **LINEAR_SCHEMAS,
    **SLACK_SCHEMAS
} 