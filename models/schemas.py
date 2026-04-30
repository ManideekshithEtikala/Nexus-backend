from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str  # Critical for identifying which "memory" to load


class ChatResponse(BaseModel):
    response: str
    session_id: str
