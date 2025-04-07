"""
Simple Context Manager for TMAI Agent.
Manages execution context during the agent's workflow.
"""

import logging
import time
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("context_manager")

class ContextManager:
    """
    Simple context manager for TMAI Agent.
    Handles creation, addition, and cleanup of execution contexts.
    """
    def __init__(self, max_contexts: int = 100):
        """
        Initialize context manager.
        
        Args:
            max_contexts: Maximum number of contexts to keep in memory
        """
        self.contexts = {}  # Dictionary mapping context_id to context data
        self.max_contexts = max_contexts
        
        logger.info(f"ContextManager initialized with max_contexts={max_contexts}")
    
    def create_context(self, channel_id: str, thread_ts: str, user_id: str, 
                      user_query: str) -> Dict[str, Any]:
        """
        Create a new execution context.
        
        Args:
            channel_id: Channel ID where the conversation is happening
            thread_ts: Thread timestamp
            user_id: User ID who initiated the query
            user_query: The actual query from the user
            
        Returns:
            The created context
        """
        context_id = f"{channel_id}:{thread_ts}"
        
        # Create new context
        context = {
            "context_id": context_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
            "user_id": user_id,
            "user_query": user_query,
            "created_at": time.time(),
            "history": [],
            "execution_results": {},
            "current_plan": {}
        }
        
        # Store context
        self.contexts[context_id] = context
        
        # Clean up if too many contexts
        if len(self.contexts) > self.max_contexts:
            self._cleanup_oldest()
        
        logger.info(f"Created context {context_id} for query: {user_query[:30]}...")
        return context
    
    def add_context(self, context_id: str, key: str, value: Any) -> bool:
        """
        Add data to an existing context.
        
        Args:
            context_id: ID of the context to update
            key: Key to update
            value: Value to set
            
        Returns:
            True if successful, False if context not found
        """
        if context_id not in self.contexts:
            logger.warning(f"Context {context_id} not found for update")
            return False
        
        # Update context
        self.contexts[context_id][key] = value
        
        logger.debug(f"Updated context {context_id} with key: {key}")
        return True
    
    def clear_context(self, context_id: str = None) -> int:
        """
        Clear one or all contexts.
        
        Args:
            context_id: Specific context to clear, or None to clear all
            
        Returns:
            Number of contexts cleared
        """
        if context_id:
            if context_id in self.contexts:
                del self.contexts[context_id]
                logger.info(f"Cleared context {context_id}")
                return 1
            return 0
        else:
            count = len(self.contexts)
            self.contexts.clear()
            logger.info(f"Cleared all {count} contexts")
            return count
    
    def get_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a context by ID.
        
        Args:
            context_id: ID of the context to retrieve
            
        Returns:
            The context dict if found, None otherwise
        """
        context = self.contexts.get(context_id)
        if not context:
            logger.warning(f"Context {context_id} not found")
        return context
    
    def _cleanup_oldest(self) -> None:
        """Remove the oldest contexts to stay within max_contexts limit"""
        # Sort contexts by creation time
        contexts_by_age = sorted(
            [(c_id, ctx["created_at"]) for c_id, ctx in self.contexts.items()],
            key=lambda x: x[1]
        )
        
        # Remove oldest to stay under limit
        to_remove = len(self.contexts) - self.max_contexts
        if to_remove <= 0:
            return
            
        for i in range(to_remove):
            if i < len(contexts_by_age):
                context_id = contexts_by_age[i][0]
                del self.contexts[context_id]
                logger.info(f"Removed old context {context_id}")

# Create a singleton instance
context_manager = ContextManager() 