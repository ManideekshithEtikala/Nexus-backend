# app/graph/nodes/persist_messages.py
"""
Stage 6 node: persist_messages

From your original main.py (step 7) + memory.compress_old_history_background:
  - Writes new_agent_messages (AIMessage / ToolMessage) generated this turn
    to the `messages` Postgres table.
  - Runs background-history compression / summary update.
  - Flushes + commits the DB session.

NOTE: The user's HumanMessage was already written to Postgres inside
`classify_importance_node` (matches your original code's early save).
This node only persists messages AFTER `turn_start_index + 1`
(i.e. everything the agent/tools produced this turn).

`compress_old_history_background` needs the FULL list of db_messages
(including the just-saved user message and the new ones) to decide what to
summarize. We re-fetch from Postgres after writing, to keep behavior
identical to your original (which used `db_messages` captured at
load-time + the newly committed user/agent messages via flush).
"""

import uuid
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import AIMessage, ToolMessage

from app.models.model import Message, ChatSession
from app.services.memory import compress_old_history_background
from app.graph.state import AgentGraphState


async def persist_messages_node(state: AgentGraphState, db: AsyncSession) -> dict:
    session_id = state["session_id"]
    session_uuid = uuid.UUID(session_id)

    turn_start_index = state.get("turn_start_index", 0)
    # +1 to skip the HumanMessage at turn_start_index (already saved in
    # classify_importance_node).
    new_agent_messages = state["messages"][turn_start_index + 1:]

    for msg in new_agent_messages:
        content_payload = str(msg.content) if msg.content else ""
        if isinstance(msg, AIMessage):
            t_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None
            db.add(Message(
                session_id=session_uuid,
                role="assistant",
                content=content_payload,
                tool_calls=t_calls,
                is_Important=False,
            ))
        elif isinstance(msg, ToolMessage):
            t_calls = {"id": getattr(msg, "tool_call_id", "legacy")}
            db.add(Message(
                session_id=session_uuid,
                role="tool",
                content=content_payload,
                tool_calls=t_calls,
                is_Important=False,
            ))

    await db.flush()

    # Re-fetch full ordered history for compression (matches original
    # `db_messages` semantics: everything saved for this session so far).
    session_result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    chat_session = session_result.scalar_one()

    msg_result = await db.execute(
        select(Message).where(Message.session_id == session_uuid).order_by(Message.created_at.asc())
    )
    db_messages = list(msg_result.scalars().all())

    await compress_old_history_background(chat_session, db_messages)
    await db.commit()

    return {}
