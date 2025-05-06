"""
Database operations for conversation management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from typing import Generator, Dict, List, Any, Optional
import logging
from dotenv import load_dotenv
from datetime import datetime
import time
import argparse
import sys

from .conversation_models import Base, Conversation, Message

load_dotenv()

logger = logging.getLogger(__name__)

# Get database URL from environment variable
# Fall back to database if DATABASE_URL isn't set
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://phattran:phatdeptrai123@localhost:5432/tmai_db')

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True  # Enable connection health checks
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# In-memory conversation cache
conversation_history = {}

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    Ensures proper handling of sessions, including error cases.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()

def init_db() -> None:
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def check_db_connection() -> bool:
    """Check if database connection is working"""
    try:
        with get_db() as db:
            db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False

def load_conversation_from_db(channel_id: str, thread_ts: str) -> Optional[List[Dict[str, Any]]]:
    """
    Load conversation history from database.
    
    Args:
        channel_id: The Slack channel ID
        thread_ts: The thread timestamp
        
    Returns:
        List of message dictionaries or None if not found
    """
    conversation_key = f"{channel_id}:{thread_ts}"
    
    # First check memory cache
    if conversation_key in conversation_history:
        return conversation_history[conversation_key]
    
    try:
        with get_db() as db:
            # Find the conversation
            conversation = db.query(Conversation).filter(
                Conversation.channel_id == channel_id,
                Conversation.thread_ts == thread_ts
            ).first()
            
            if not conversation:
                return None
            
            # Get all messages for this conversation
            messages = db.query(Message).filter(
                Message.conversation_id == conversation.id
            ).order_by(Message.timestamp).all()
            
            # Format messages
            formatted_messages = []
            for msg in messages:
                message_dict = {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": float(msg.timestamp),  # Ensure timestamp is float
                    "message_ts": msg.message_ts,
                    "metadata": msg.meta_data or {}
                }
                formatted_messages.append(message_dict)
            
            # Cache in memory
            conversation_history[conversation_key] = formatted_messages
            
            return formatted_messages
            
    except Exception as e:
        logger.error(f"Error loading conversation from database: {str(e)}")
        return None

def save_conversation_to_db(channel_id: str, thread_ts: str, messages: List[Dict[str, Any]]) -> bool:
    """
    Save conversation history to database.
    
    Args:
        channel_id: The Slack channel ID
        thread_ts: The thread timestamp
        messages: List of message dictionaries to save
        
    Returns:
        True if successful, False otherwise
    """
    conversation_key = f"{channel_id}:{thread_ts}"
    
    try:
        with get_db() as db:
            # Find or create conversation
            conversation = db.query(Conversation).filter(
                Conversation.channel_id == channel_id,
                Conversation.thread_ts == thread_ts
            ).first()
            
            if not conversation:
                conversation = Conversation(
                    channel_id=channel_id,
                    thread_ts=thread_ts,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                db.add(conversation)
                db.flush()  # Get the ID without committing
            else:
                conversation.updated_at = datetime.utcnow()
            
            # Delete existing messages
            db.query(Message).filter(
                Message.conversation_id == conversation.id
            ).delete()
            
            # Add new messages
            for msg in messages:
                # Ensure timestamp is a float
                timestamp = msg["timestamp"] if isinstance(msg["timestamp"], (int, float)) else time.time()
                
                message = Message(
                    conversation_id=conversation.id,
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=timestamp,  # Use the float timestamp
                    message_ts=msg.get("message_ts"),
                    meta_data=msg.get("metadata", {})
                )
                db.add(message)
            
            db.commit()
            
            # Update memory cache
            conversation_history[conversation_key] = messages
            
            return True
            
    except Exception as e:
        logger.error(f"Error saving conversation to database: {str(e)}")
        return False

def cleanup_old_conversations(hours: int = 24) -> int:
    """
    Remove conversations older than the specified number of hours from the database.
    
    Args:
        hours: Number of hours old a conversation must be to be deleted
        
    Returns:
        Number of conversations removed
    """
    from sqlalchemy import delete
    from sqlalchemy.orm import Session
    from ops_conversation_db.conversation_models import Conversation
    from datetime import datetime, timedelta
    
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    logger.info(f"Cleaning up conversations older than {cutoff_time}")
    
    try:
        with get_db() as db:
            # Find conversations to delete
            old_conversations = db.query(Conversation).filter(
                Conversation.updated_at < cutoff_time
            ).all()
            
            if not old_conversations:
                logger.info("No old conversations found to clean up")
                return 0
                
            # Get IDs for logging
            conversation_ids = [conv.id for conv in old_conversations]
            count = len(conversation_ids)
            
            # Delete messages for old conversations first to avoid FK constraint errors
            from ops_conversation_db.conversation_models import Message
            db.query(Message).filter(
                Message.conversation_id.in_(conversation_ids)
            ).delete(synchronize_session=False)
            # Delete conversations
            db.query(Conversation).filter(
                Conversation.id.in_(conversation_ids)
            ).delete(synchronize_session=False)
            # Commit deletions
            db.commit()
            logger.info(f"Cleaned up {count} old conversations")
            return count
            
    except Exception as e:
        logger.error(f"Error cleaning up old conversations: {str(e)}")
        return 0

def view_conversations(limit: int = 2, channel_id: str = None, thread_ts: str = None) -> None:
    """
    View conversations and their messages from the database.
    
    Args:
        limit: Maximum number of conversations to display
        channel_id: Optional filter for a specific channel
        thread_ts: Optional filter for a specific thread timestamp
    """
    from sqlalchemy import and_, desc
    
    try:
        with get_db() as db:
            # Build query with optional filters
            query = db.query(Conversation).order_by(desc(Conversation.created_at))
            
            if channel_id:
                query = query.filter(Conversation.channel_id == channel_id)
            
            if thread_ts:
                query = query.filter(Conversation.thread_ts == thread_ts)
                
            # Get conversations
            conversations = query.limit(limit).all()
            
            if not conversations:
                print("No conversations found in the database with the specified filters.")
                return
                
            print(f"\n===== CONVERSATIONS ({len(conversations)}) =====")
            for conv in conversations:
                print(f"\nID: {conv.id}")
                print(f"Channel ID: {conv.channel_id}")
                print(f"Thread TS: {conv.thread_ts}")
                print(f"Created At: {conv.created_at}")
                print(f"Updated At: {conv.updated_at}")
                print(f"Meta Data: {conv.meta_data}")
                
                # Get messages for this conversation
                messages = db.query(Message).filter(
                    Message.conversation_id == conv.id
                ).order_by(Message.timestamp).all()
                
                if messages:
                    print(f"\n  --- MESSAGES ({len(messages)}) ---")
                    for msg in messages:
                        print(f"  Message ID: {msg.id}")
                        print(f"  Role: {msg.role}")
                        # Show a shorter preview of content
                        content_preview = msg.content[:60] + "..." if len(msg.content) > 60 else msg.content
                        print(f"  Content: {content_preview}")
                        print(f"  Timestamp: {msg.timestamp}")
                        if msg.message_ts:
                            print(f"  Message TS: {msg.message_ts}")
                        if msg.meta_data:
                            print(f"  Meta Data: {msg.meta_data}")
                        print("  " + "-"*40)
                else:
                    print("  No messages found for this conversation.")
                
                print("="*50)
    
    except Exception as e:
        print(f"Error viewing conversations: {str(e)}")
        import traceback
        traceback.print_exc()

# Allow running this file directly to view conversations
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description='Conversation database operations')
    parser.add_argument('--cleanup', type=int, help='Clean up conversations older than N hours')
    parser.add_argument('--view', action='store_true', help='View recent conversations')
    parser.add_argument('--channel', type=str, help='Channel ID for viewing specific conversations')
    parser.add_argument('--thread', type=str, help='Thread TS for viewing specific conversation')
    parser.add_argument('--limit', type=int, default=5, help='Limit for number of conversations to view')
    parser.add_argument('--init', action='store_true', help='Initialize database tables')
    
    args = parser.parse_args()
    
    # Always try to initialize the database to ensure tables exist
    try:
        logger.info("Ensuring database tables are initialized...")
        init_db()
        logger.info("Database tables ready")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}")
        sys.exit(1)
    
    if args.cleanup:
        try:
            count = cleanup_old_conversations(hours=args.cleanup)
            print(f"Cleaned up {count} conversations older than {args.cleanup} hours")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            sys.exit(1)
    
    elif args.view:
        try:
            view_conversations(
                limit=args.limit,
                channel_id=args.channel,
                thread_ts=args.thread
            )
        except Exception as e:
            logger.error(f"Error viewing conversations: {e}")
            sys.exit(1)
    
    elif args.init:
        # We've already initialized above
        print("Database tables initialized successfully")
    
    else:
        parser.print_help() 