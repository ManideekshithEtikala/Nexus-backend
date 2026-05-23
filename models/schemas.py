from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    message: str
    session_id: str  # Critical for identifying which "memory" to load
    image: Optional[str] = None  # Base64 encoded image payload


class ChatResponse(BaseModel):
    response: str
    session_id: str
