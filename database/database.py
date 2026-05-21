import os
import logging
import ssl
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from core.config import settings

logger = logging.getLogger(__name__)

# Configure SSL for asyncpg based on reference
connect_args = {
    "server_settings": {"jit": "off"},  # Disable JIT for short queries
    "command_timeout": 60,
}

if settings.DATABASE_SSL and settings.DATABASE_SSL != "disable":
    if settings.DATABASE_SSL == "require":
        # For development, we'll disable SSL verification (as per reference)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        connect_args["ssl"] = ssl_context

# Async SQLAlchemy engine
async_engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,  # 5 per worker × 2 workers = 10 total (optimized for Aiven free tier)
    max_overflow=5,  # Allow burst to 10 per worker max
    pool_recycle=1800,  # Recycle every 30 min (Aiven kills idle at 5min)
    pool_timeout=30,  # Wait max 30s for a connection
    connect_args=connect_args,
)


async def initialize_database_schemas():
    """
    Asynchronously bootstraps the database.
    Creates the 'conversation_summaries' and standard indexing table if they do not exist,
    preventing downstream crashes when conversation summaries are processed.
    """
    from sqlalchemy import text
    try:
        print("[Database] Starting database schema verification & bootstrapping...")
        async with async_engine.begin() as conn:
            # Create conversation_summaries table
            await conn.execute(text("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id SERIAL PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    summary_text TEXT NOT NULL,
                    covered_message_count INTEGER NOT NULL,
                    last_message_id INTEGER,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """))
            
            # Create index on session_id and created_at
            await conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_summaries_session_created 
                ON conversation_summaries(session_id, created_at DESC);
            """))
            
        print("[Database] Database schema bootstrapped successfully.")
    except Exception as e:
        print(f"[Database] Error bootstrapping schema: {e}")


def get_session_history(session_id: str) -> SQLChatMessageHistory:
    """
    Returns a SQLChatMessageHistory for the given session_id.
    The table 'message_store' is created automatically on first use.
    Each unique session_id gets its own isolated memory window.
    """
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine,
        table_name="message_store",
        async_mode=True,
    )
