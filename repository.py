from sqlalchemy.orm import Session
from models import Conversation, Message
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ConversationRepository:
    """Repository for conversation-related database operations"""

    @staticmethod
    def create_conversation(db: Session, channel_id: str, thread_ts: str, metadata: Optional[Dict] = None) -> Conversation:
        """Create a new conversation"""
        conversation = Conversation(
            id=f"{channel_id}:{thread_ts}",
            channel_id=channel_id,
            thread_ts=thread_ts,
            metadata=metadata
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)
        return conversation

    @staticmethod
    def get_conversation(db: Session, channel_id: str, thread_ts: str) -> Optional[Conversation]:
        """Get a conversation by channel_id and thread_ts"""
        return db.query(Conversation).filter(
            Conversation.id == f"{channel_id}:{thread_ts}"
        ).first()

    @staticmethod
    def add_message(
        db: Session,
        conversation_id: str,
        role: str,
        content: str,
        timestamp: float,
        message_ts: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Message:
        """Add a message to a conversation"""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            timestamp=timestamp,
            message_ts=message_ts,
            metadata=metadata
        )
        db.add(message)
        db.commit()
        db.refresh(message)
        return message

    @staticmethod
    def get_conversation_messages(
        db: Session,
        channel_id: str,
        thread_ts: str,
        limit: Optional[int] = None
    ) -> List[Message]:
        """Get messages for a conversation"""
        query = db.query(Message).join(Conversation).filter(
            Conversation.channel_id == channel_id,
            Conversation.thread_ts == thread_ts
        ).order_by(Message.timestamp.asc())

        if limit:
            query = query.limit(limit)

        return query.all()

    @staticmethod
    def clean_old_conversations(db: Session, hours: int = 24) -> int:
        """Delete conversations older than specified hours"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        old_conversations = db.query(Conversation).filter(
            Conversation.updated_at < cutoff_time
        ).all()

        count = len(old_conversations)
        for conv in old_conversations:
            db.delete(conv)
        
        db.commit()
        return count

    @staticmethod
    def update_conversation_metadata(
        db: Session,
        channel_id: str,
        thread_ts: str,
        metadata: Dict[str, Any]
    ) -> Optional[Conversation]:
        """Update conversation metadata"""
        conversation = ConversationRepository.get_conversation(db, channel_id, thread_ts)
        if conversation:
            conversation.metadata = metadata
            db.commit()
            db.refresh(conversation)
        return conversation 