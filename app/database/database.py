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
# Make sure DATABASE_URL uses postgresql+asyncpg:// in your .env

engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    pool_size=10,
    max_overflow=5,
    connect_args={"ssl": "require"}  # ← Fixed for Supabase
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