import os
import ssl
from typing import AsyncGenerator
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.engine.url import URL

load_dotenv()

# 1. Pull the explicit credentials from your updated .env file
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 12070))
DB_NAME = os.getenv("DB_NAME")

# 2. Programmatically build the exact string SQLAlchemy needs
DATABASE_URL = URL.create(
    drivername="postgresql+asyncpg",
    username=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST.strip() if DB_HOST else None,  # Safely removes any hidden trailing spaces
    port=DB_PORT,
    database=DB_NAME
)

# 3. Securely set up the SSL context required by Aiven Cloud
# ssl_context = ssl.create_default_context()
# ssl_context.check_hostname = False
# ssl_context.verify_mode = ssl.CERT_NONE

# 4. Create the engine passing your custom SSL arguments
engine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True,
    pool_size=10,
    max_overflow=5,
    connect_args={"ssl": False}  # Injects SSL parameters directly to asyncpg
)

# 5. Create the session factory
async_session_local = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# 6. Global Base class for your tables to inherit from
class Base(DeclarativeBase):
    pass

# 7. FastAPI dependency injection context generator
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Yields a database session to your API routes.
    Automatically handles commits, rollbacks on errors, and closing connections.
    """
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
    """Call this on application startup to create tables automatically."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)