import uuid
from datetime import datetime
from sqlalchemy import String, ForeignKey, DateTime, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import List
from pydantic import BaseModel, Field
from app.database import Base

from sqlalchemy import String, DateTime, Text # Import Text
# ... other imports ...

class ChatSession(Base):
    '''Represents a conversation session, which can contain multiple messages.'''
    __tablename__ = "chat_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    title: Mapped[str] = mapped_column(String(255), default="New Conversation")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True) # 👈 Add this to store the running summary!
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    messages: Mapped[list["Message"]] = relationship(back_populates="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"))
    role: Mapped[str] = mapped_column(String(50)) # user, assistant, tool
    content: Mapped[str] = mapped_column(String)
    tool_calls: Mapped[dict | None] = mapped_column(JSON, nullable=True) # Stores agent thoughts
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    is_Important: Mapped[bool] = mapped_column(default=False) # Flag for important messages
    session: Mapped["ChatSession"] = relationship(back_populates="messages")
    
