#!/usr/bin/env python3
"""
Test script to initialize the conversation database tables
"""

import os
import sys
import logging
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, ForeignKey
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("db_init_test")

# Define models similar to conversation_models.py
Base = declarative_base()

class Conversation(Base):
    """Model for storing conversation threads"""
    __tablename__ = 'conversations'

    id = Column(String, primary_key=True)
    channel_id = Column(String, nullable=False)
    thread_ts = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta_data = Column(JSON, nullable=True)

class Message(Base):
    """Model for storing individual messages in conversations"""
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    conversation_id = Column(String, ForeignKey('conversations.id'), nullable=False)
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)
    timestamp = Column(Float, nullable=False)
    message_ts = Column(String, nullable=True)
    meta_data = Column(JSON, nullable=True)

def init_db():
    """Initialize database tables"""
    # Get database URL from environment or use a default
    database_url = os.environ.get(
        'DATABASE_URL', 
        'postgresql://phattran:phatdeptrai123@localhost:5432/tmai_db'
    )
    
    # Update for Docker container
    if 'localhost' in database_url and os.environ.get('POSTGRES_HOST'):
        database_url = database_url.replace(
            'localhost', 
            os.environ.get('POSTGRES_HOST')
        )
    
    logger.info(f"Connecting to database: {database_url}")
    
    # Create engine and tables
    engine = create_engine(database_url)
    
    try:
        # First test connection
        connection = engine.connect()
        connection.close()
        logger.info("Database connection successful")
        
        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing database initialization...")
    
    # Override the database host if needed
    if len(sys.argv) > 1:
        os.environ['POSTGRES_HOST'] = sys.argv[1]
        
    success = init_db()
    
    if success:
        logger.info("✅ Test completed successfully")
        sys.exit(0)
    else:
        logger.error("❌ Test failed")
        sys.exit(1) 