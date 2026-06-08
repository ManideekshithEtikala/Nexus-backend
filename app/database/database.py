import os
from typing import AsyncGenerator
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

DATABASE_URL: str = os.environ["DATABASE_URL"]

# Ensure the URL uses the correct asyncpg dialect scheme
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+asyncpg://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)

engine = create_async_engine(
    DATABASE_URL,
    echo=False,           # Turn off verbose echo in production to preserve log space
    future=True,
    pool_size=4,          # Safe ceiling for Supabase free-tier connection limits
    max_overflow=2,
    pool_pre_ping=True,   # Optimistically tests connections before running queries
    pool_recycle=1800,    # Recycle connections every 30 minutes to drop dead sockets
    connect_args={
        "ssl": "require",          # Let asyncpg negotiate the SSL layer securely without hardcoded contexts
        "timeout": 30,             # Protect against cold-start network drops
        "command_timeout": 30,
        "statement_cache_size": 0   # CRITICAL: Disables prepared statements so you can use poolers safely
    }
)

async_session_local = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

class Base(DeclarativeBase):
    pass

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_local() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)