"""
Linear database module for TMAI Agent.
Handles operations with Linear API and semantic search.
"""

from .linear_client import (
    LinearClient,
    LinearError,
    LinearAuthError,
    LinearNotFoundError,
    LinearValidationError
)

from .linear_rag_embeddings import (
    semantic_search,
    store_issue_embedding,
    store_project_embedding,
    store_comment_embedding,
    get_embedding
)

from .db_pool import get_db_connection

__all__ = [
    # Classes
    'LinearClient',
    
    # Exceptions
    'LinearError',
    'LinearAuthError',
    'LinearNotFoundError',
    'LinearValidationError',
    
    # Functions
    'semantic_search',
    'store_issue_embedding',
    'store_project_embedding',
    'store_comment_embedding',
    'get_embedding',
    'get_db_connection'
]
