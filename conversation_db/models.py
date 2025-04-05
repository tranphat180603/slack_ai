from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Conversation(Base):
    """Model for storing conversation threads"""
    __tablename__ = 'conversations'

    id = Column(String, primary_key=True)  # channel_id:thread_ts
    channel_id = Column(String, nullable=False)
    thread_ts = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = Column(JSON, nullable=True)  # For storing any additional metadata

    def __init__(self, channel_id, thread_ts, **kwargs):
        """Initialize with composite id from channel_id and thread_ts"""
        super().__init__(**kwargs)
        self.id = f"{channel_id}:{thread_ts}"
        self.channel_id = channel_id
        self.thread_ts = thread_ts

    # Relationship with messages
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    """Model for storing individual messages in conversations"""
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    message_ts = Column(String, nullable=True)  # Slack message timestamp
    meta_data = Column(JSON, nullable=True)  # For storing any additional metadata

    # Relationship with conversation
    conversation = relationship("Conversation", back_populates="messages")

    class Config:
        orm_mode = True 