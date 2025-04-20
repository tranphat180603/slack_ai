# test_conversation.py
"""
Tests for cleanup_old_conversations to ensure conversations and messages
are deleted without foreign key errors.
"""
try:
    import pytest
except ImportError:
    class _DummyPytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def wrapper(f):
                return f
            return wrapper
    pytest = _DummyPytest()
import time
from datetime import datetime, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import ops_conversation_db.conversation_db as db
import ops_conversation_db.conversation_models as models


@pytest.fixture(autouse=True)
def in_memory_db(monkeypatch):
    """
    Set up an in-memory SQLite database for testing and patch SessionLocal.
    """
    # Clear in-memory cache
    db.conversation_history.clear()
    # Create SQLite in-memory engine
    engine = create_engine("sqlite:///:memory:", echo=False)
    # Create tables
    models.Base.metadata.create_all(bind=engine)
    # Create a new session factory bound to the in-memory engine
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    # Patch the SessionLocal used by conversation_db
    monkeypatch.setattr(db, "SessionLocal", SessionLocal)
    return engine


def test_cleanup_old_conversation_and_messages_removed():
    # Insert a conversation older than cutoff and one associated message
    cutoff_hours = 1
    with db.get_db() as session:
        conv = models.Conversation(channel_id="ch1", thread_ts="ts1")
        # Manually set timestamps to be in the past beyond cutoff
        past_time = datetime.utcnow() - timedelta(hours=cutoff_hours + 1)
        conv.created_at = past_time
        conv.updated_at = past_time
        session.add(conv)
        session.flush()
        msg = models.Message(
            conversation_id=conv.id,
            role="user",
            content="hello",
            timestamp=time.time(),
            message_ts="mts1",
        )
        session.add(msg)
    # Ensure data was inserted
    with db.get_db() as session:
        assert session.query(models.Conversation).count() == 1
        assert session.query(models.Message).count() == 1

if __name__ == "__main__":
    import logging, sys
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from datetime import datetime, timedelta
    # Configure logging to show info from ops_conversation_db
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout
    )
    logger = logging.getLogger(__name__)

    # Clear any in-memory cache and set up in-memory SQLite DB
    db.conversation_history.clear()
    engine = create_engine("sqlite:///:memory:", echo=False)
    models.Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db.SessionLocal = SessionLocal

    cutoff_hours = 1
    # Test 1: old conversation should be removed
    logger.info("--- Test 1: Removing old conversation and messages ---")
    try:
        with db.get_db() as session:
            conv = models.Conversation(channel_id="ch1", thread_ts="ts1")
            past = datetime.utcnow() - timedelta(hours=cutoff_hours + 1)
            conv.created_at = past
            conv.updated_at = past
            session.add(conv)
            session.flush()
            msg = models.Message(
                conversation_id=conv.id,
                role="user",
                content="hello",
                timestamp=time.time(),
                message_ts="mts1"
            )
            session.add(msg)
        with db.get_db() as session:
            c_before = session.query(models.Conversation).count()
            m_before = session.query(models.Message).count()
        logger.info(f"Before cleanup: conversations={c_before}, messages={m_before}")
        removed = db.cleanup_old_conversations(hours=cutoff_hours)
        logger.info(f"cleanup_old_conversations returned: {removed}")
        with db.get_db() as session:
            c_after = session.query(models.Conversation).count()
            m_after = session.query(models.Message).count()
        logger.info(f"After cleanup: conversations={c_after}, messages={m_after}")
        assert removed == 1 and c_after == 0 and m_after == 0
        logger.info("Test 1 passed: old conversation removed successfully.")
    except Exception:
        logger.exception("Test 1 failed")
        sys.exit(1)

    # Test 2: recent conversation should stay
    logger.info("--- Test 2: Retaining recent conversation and messages ---")
    try:
        with db.get_db() as session:
            conv = models.Conversation(channel_id="ch2", thread_ts="ts2")
            session.add(conv)
            session.flush()
            msg = models.Message(
                conversation_id=conv.id,
                role="assistant",
                content="world",
                timestamp=time.time(),
                message_ts="mts2"
            )
            session.add(msg)
        with db.get_db() as session:
            c_before = session.query(models.Conversation).count()
            m_before = session.query(models.Message).count()
        logger.info(f"Before cleanup: conversations={c_before}, messages={m_before}")
        removed = db.cleanup_old_conversations(hours=cutoff_hours)
        logger.info(f"cleanup_old_conversations returned: {removed}")
        with db.get_db() as session:
            c_after = session.query(models.Conversation).count()
            m_after = session.query(models.Message).count()
        logger.info(f"After cleanup: conversations={c_after}, messages={m_after}")
        assert removed == 0 and c_after == 1 and m_after == 1
        logger.info("Test 2 passed: recent conversation retained successfully.")
    except Exception:
        logger.exception("Test 2 failed")
        sys.exit(1)

    logger.info("All tests passed. No FK errors encountered.")
    sys.exit(0)
    # Perform cleanup
    removed = db.cleanup_old_conversations(hours=cutoff_hours)
    assert removed == 1
    # Both conversation and message should be gone
    with db.get_db() as session:
        assert session.query(models.Conversation).count() == 0
        assert session.query(models.Message).count() == 0


def test_cleanup_keeps_recent_conversation():
    # Insert a recent conversation within cutoff and one associated message
    cutoff_hours = 1
    with db.get_db() as session:
        conv = models.Conversation(channel_id="ch2", thread_ts="ts2")
        # Leave timestamps at default (now)
        session.add(conv)
        session.flush()
        msg = models.Message(
            conversation_id=conv.id,
            role="assistant",
            content="world",
            timestamp=time.time(),
            message_ts="mts2",
        )
        session.add(msg)
    # Ensure data was inserted
    with db.get_db() as session:
        assert session.query(models.Conversation).count() == 1
        assert session.query(models.Message).count() == 1
    # Perform cleanup
    removed = db.cleanup_old_conversations(hours=cutoff_hours)
    assert removed == 0
    # Conversation and message should still exist
    with db.get_db() as session:
        assert session.query(models.Conversation).count() == 1
        assert session.query(models.Message).count() == 1