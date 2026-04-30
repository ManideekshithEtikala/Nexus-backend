from urllib.parse import quote_plus
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory

# URL-encode the password so @ in "Manideekshith@11" doesn't break the URL parser
_password = quote_plus("Manideekshith@11")
DB_URL = f"postgresql+asyncpg://postgres:{_password}@localhost:5432/chat_bot"

# Async SQLAlchemy engine
async_engine = create_async_engine(DB_URL, pool_pre_ping=True, echo=False)


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
