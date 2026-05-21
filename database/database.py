import os
from urllib.parse import quote_plus
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Load DB_URL from environment or fallback to default local postgres setup
_password = quote_plus("Manideekshith@11")
DEFAULT_DB_URL = f"postgresql+asyncpg://postgres:{_password}@localhost:5432/chat_bot"
DB_URL = os.getenv("DATABASE_URL", DEFAULT_DB_URL).strip()

# Async SQLAlchemy engine
async_engine = create_async_engine(DB_URL, pool_pre_ping=True, echo=False)


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
