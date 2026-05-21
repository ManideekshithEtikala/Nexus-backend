import os
from pathlib import Path
from urllib.parse import quote_plus
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API Keys
    API_KEY: str
    GEMINI_API_KEY: str
    
    # Database Configuration
    DATABASE_NAME: str
    DATABASE_USER_NAME: str
    DATABASE_PASS: str
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_SSL: str = "require"
    
    # Security info
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480 

    # Admin Credentials
    ADMIN_CODE: str = "admin"
    ADMIN_PASSWORD: str = "admin"
    
    # Vector Memory Configuration
    EMBEDDING_PROVIDER: str = "google"
    EMBEDDING_MODEL: str = "text-embedding-004"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "nexus-memory"
    
    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379"

    # CORS
    CORS_ORIGINS: str = "*"
    STORAGE_DIR_NAME: str = "storage"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse comma-separated CORS_ORIGINS into a list."""
        if not self.CORS_ORIGINS or self.CORS_ORIGINS == "*":
            return ["*"]
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def DATABASE_URL(self) -> str:
        """Return a database URL with asyncpg driver."""
        encoded_pass = quote_plus(self.DATABASE_PASS)
        return (
            f"postgresql+asyncpg://{self.DATABASE_USER_NAME}:"
            f"{encoded_pass}@{self.DATABASE_HOST}:"
            f"{self.DATABASE_PORT}/{self.DATABASE_NAME}"
        )

    class Config:
        env_file = ".env"
        extra = "ignore" 

settings = Settings()
