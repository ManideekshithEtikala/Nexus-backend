# app/services/memory.py
import os
import uuid
from typing import List
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate

from app.models.model import ChatSession, Message
from app.core.schema import AgentState, MemoryTaggingSchema

api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1, api_key=SecretStr(api_key))

# =====================================================================
# 🧠 MEMORY CLASSIFIER (Restored!)
# =====================================================================
tagging_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an elite database state memory optimization controller. Evaluate the structural value of the text."),
    ("human", "Analyze the following content and extract if it should be marked as permanently important:\n\nContent: {content}"),
])
classifier_chain = tagging_prompt | llm.with_structured_output(MemoryTaggingSchema)

async def get_or_create_session(db: AsyncSession, session_id: str, first_message: str) -> ChatSession:
    """Fetches an existing chat session from PostgreSQL, or creates a new one."""
    session_result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    chat_session = session_result.scalar_one_or_none()
    if not chat_session:
        chat_session = ChatSession(id=session_id, title=first_message[:30], summary="")
        db.add(chat_session)
        await db.commit()
    return chat_session

async def populate_state_context(db: AsyncSession, state: AgentState) -> list[Message]:
    """Rebuilds the AI's LangChain message array from PostgreSQL history."""
    msg_result = await db.execute(select(Message).where(Message.session_id == uuid.UUID(state.session_id)).order_by(Message.created_at.asc()))
    db_messages = list(msg_result.scalars().all())

    important_msgs = [msg for msg in db_messages if msg.is_Important]
    WINDOW_SIZE = 6
    recent_db_messages = db_messages[-WINDOW_SIZE:] if len(db_messages) > WINDOW_SIZE else db_messages

    combined_messages = list(dict.fromkeys(important_msgs + list(recent_db_messages)))
    combined_messages.sort(key=lambda x: x.created_at)

    for msg in combined_messages:
        if msg.role == "user":
            state.messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            state.messages.append(AIMessage(content=msg.content or "", tool_calls=msg.tool_calls or []))
        elif msg.role == "tool":
            t_id = msg.tool_calls.get("id", "legacy") if isinstance(msg.tool_calls, dict) else "legacy"
            state.messages.append(ToolMessage(content=msg.content, tool_call_id=t_id))

    return db_messages

async def compress_old_history_background(chat_session: ChatSession, db_messages: list):
    """Background task: Summarizes old messages so the DB doesn't overload the LLM context window."""
    WINDOW_SIZE = 6
    if len(db_messages) <= WINDOW_SIZE: return
    
    older_messages = db_messages[:-WINDOW_SIZE]
    unimportant_older_messages = [m for m in older_messages if not m.is_Important]
    if not unimportant_older_messages: return

    formatted_old_chat = "\n".join([f"{m.role}: {m.content}" for m in unimportant_older_messages])
    summary_prompt = f"Merge these transcript lines into the existing context.\nExisting: {chat_session.summary}\nNew:\n{formatted_old_chat}"
    try:
        summary_response = await llm.ainvoke(summary_prompt)
        chat_session.summary = str(summary_response.content)
    except Exception as e:
        print(f"Summary failure: {e}")