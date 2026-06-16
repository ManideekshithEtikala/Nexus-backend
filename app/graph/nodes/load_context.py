# app/graph/nodes/load_context.py
"""
Stage 4 node: load_context

Combines (from your original code):
  - memory.get_or_create_session
  - memory.populate_state_context

This is the entry node. It is called ONCE per graph invocation (per HTTP
request), even though the checkpointer also persists graph state between
turns. Why both?

  - The CHECKPOINTER persists the *graph's* state (messages, iteration,
    behaviour, etc.) keyed by `thread_id` (= session_id). This is what makes
    interrupt()/resume work, and on a 2nd+ turn it will already contain the
    previous turn's messages.
  - Your Postgres `ChatSession` / `Message` tables remain the source of truth
    for: (a) the UI's chat history display, (b) the running `summary` text,
    (c) the `is_Important` flag bookkeeping.

On the FIRST turn for a session (checkpointer has nothing yet), this node
also seeds `messages` from Postgres history (if any existed before the
migration, or if you want Postgres to remain authoritative). On SUBSEQUENT
turns, the checkpointer will have already restored `messages`, so we must be
careful not to duplicate history.

STRATEGY USED HERE (simple + safe):
  - Always fetch chat_session (get_or_create).
  - Only seed `messages` from Postgres if the checkpointer's restored state
    has an EMPTY messages list (i.e. this is truly the first turn for this
    thread_id). Otherwise, leave `messages` alone — the checkpointer already
    has it.
  - Always set `current_summary` from chat_session.summary (Postgres is
    authoritative for the rolling summary, updated by persist_messages).
"""

import uuid
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

from app.models.model import ChatSession, Message
from ..state import AgentGraphState
from app.core.schema import BehaviourPattern

# Must match the id used in inject_permanent_facts.py
SYSTEM_MESSAGE_ID = "system-prompt"


async def get_or_create_session(db: AsyncSession, session_id: str, first_message: str) -> ChatSession:
    """Fetches an existing chat session from PostgreSQL, or creates a new one."""
    session_result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    chat_session = session_result.scalar_one_or_none()
    if not chat_session:
        chat_session = ChatSession(id=session_id, title=first_message[:30], summary="")
        db.add(chat_session)
        await db.commit()
    return chat_session


def _row_to_message(msg: Message):
    """Converts a Postgres Message row into a LangChain message object."""
    if msg.role == "user":
        return HumanMessage(content=msg.content)
    elif msg.role == "assistant":
        return AIMessage(content=msg.content or "", tool_calls=msg.tool_calls or [])
    elif msg.role == "tool":
        t_id = msg.tool_calls.get("id", "legacy") if isinstance(msg.tool_calls, dict) else "legacy"
        return ToolMessage(content=msg.content, tool_call_id=t_id)
    return None


async def load_context_node(state: AgentGraphState, db: AsyncSession, user_message: str) -> dict:
    """
    Graph entry node.

    `db` and `user_message` are injected via a closure/partial when building
    the graph invocation (see agent_graph.py `build_graph` and main.py),
    since LangGraph nodes only receive `state` (and optionally `config`) by
    default — extra runtime dependencies are passed through a wrapper.
    """
    session_id = state["session_id"]
    session_uuid = uuid.UUID(session_id)

    chat_session = await get_or_create_session(db, session_id, user_message)

    update: dict = {
        "current_summary": chat_session.summary or "",
    }

    # TypedDict has no Pydantic defaults. Seed `current_behaviour` on the
    # very first turn (checkpointer has nothing yet); on later turns the
    # checkpointer already restored whatever value change_behaviour_profile
    # last set.
    if "current_behaviour" not in state:
        update["current_behaviour"] = BehaviourPattern.CASUAL_PRODUCTIVITY.value

    if "iteration" not in state:
        update["iteration"] = 0

    # Only seed history from Postgres if the checkpointer gave us an empty
    # `messages` list (i.e. first-ever turn for this thread_id).
    existing_messages = state.get("messages", [])
    if not existing_messages:
        msg_result = await db.execute(
            select(Message)
            .where(Message.session_id == session_uuid)
            .order_by(Message.created_at.asc())
        )
        db_messages = list(msg_result.scalars().all())

        important_msgs = [m for m in db_messages if m.is_Important]
        WINDOW_SIZE = 6
        recent_db_messages = db_messages[-WINDOW_SIZE:] if len(db_messages) > WINDOW_SIZE else db_messages

        combined = list(dict.fromkeys(important_msgs + list(recent_db_messages)))
        combined.sort(key=lambda x: x.created_at)

        seeded = []
        for msg in combined:
            converted = _row_to_message(msg)
            if converted is not None:
                seeded.append(converted)

        # Placeholder SystemMessage at position 0 with a STABLE id.
        # inject_permanent_facts_node will later replace its content
        # in-place (same id => same position, see that file's docstring).
        update["messages"] = [SystemMessage(content="", id=SYSTEM_MESSAGE_ID)] + seeded

    return update
