from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import os
from typing import Generator
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Get database URL from environment variable
# Fall back to linear_rag database if DATABASE_URL isn't set
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://postgres:phatdeptrai123@localhost:5432/linear_rag')

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
    from models import Base
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

def cleanup_old_conversations(hours: int = 24) -> int:
    """
    Remove conversations older than the specified number of hours from the database.
    
    Args:
        hours: Number of hours old a conversation must be to be deleted
        
    Returns:
        Number of conversations removed
    """
    from sqlalchemy import delete
    from models import Conversation
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
            
            # Delete conversations (cascade will delete messages too)
            db.query(Conversation).filter(
                Conversation.updated_at < cutoff_time
            ).delete()
            
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
    from models import Conversation, Message
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
    import sys
    
    # Simple command line argument parsing
    channel = None
    thread = None
    cleanup = False
    cleanup_hours = 24
    
    # Check for arguments
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.startswith("--channel="):
                channel = arg.split("=")[1]
            elif arg.startswith("--thread="):
                thread = arg.split("=")[1]
            elif arg.startswith("--cleanup"):
                cleanup = True
                if "=" in arg:
                    try:
                        cleanup_hours = int(arg.split("=")[1])
                    except ValueError:
                        print(f"Invalid hours value: {arg.split('=')[1]}, using default 24 hours")
            elif arg.startswith("--help"):
                print("Usage: python database.py [--channel=CHANNEL_ID] [--thread=THREAD_TS] [--cleanup[=HOURS]]")
                print("  --channel=CHANNEL_ID   Filter conversations by channel ID")
                print("  --thread=THREAD_TS     Filter conversations by thread timestamp")
                print("  --cleanup[=HOURS]      Clean up conversations older than specified hours (default: 24)")
                print("  --help                 Show this help message")
                sys.exit(0)
    
    # Run cleanup if requested
    if cleanup:
        removed = cleanup_old_conversations(hours=cleanup_hours)
        print(f"Cleaned up {removed} conversations older than {cleanup_hours} hours")
    else:
        # Otherwise view conversations
        view_conversations(channel_id=channel, thread_ts=thread) 