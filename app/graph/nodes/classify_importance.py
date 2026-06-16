# app/graph/nodes/classify_importance.py
"""
Stage 5 node: classify_importance

From your original main.py:
    user_classification = await classifier_chain.ainvoke({"content": payload.message})
    state.is_user_msg_important = user_classification.is_Important
    state.messages.append(HumanMessage(content=payload.message))
    db.add(Message(session_id=..., role="user", content=..., is_Important=...))

This node:
  1. Classifies the incoming user message's importance.
  2. Appends the HumanMessage to graph state (via add_messages reducer).
  3. Writes the user message row to Postgres immediately (matches your
     original "save user message right away" behavior).
  4. Records `turn_start_index` = current message count BEFORE this turn's
     new AI/tool messages get added — used later by persist_messages to know
     what's "new" this turn.

Note on `turn_start_index`: we compute it as len(state["messages"]) BEFORE
appending the new HumanMessage via the reducer. Since add_messages appends
deterministically, after this node runs the HumanMessage will be at index
`turn_start_index`. persist_messages will slice from `turn_start_index + 1`
onward to get only the assistant/tool messages generated THIS turn.
"""

import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_core.messages import HumanMessage

from app.models.model import Message
from app.services.memory import classifier_chain
from app.graph.state import AgentGraphState


async def classify_importance_node(state: AgentGraphState, db: AsyncSession, user_message: str) -> dict:
    session_uuid = uuid.UUID(state["session_id"])

    try:
        classification = await classifier_chain.ainvoke({"content": user_message})
        is_important = classification.is_Important
    except Exception:
        is_important = False

    # Save user message to Postgres immediately (matches original behavior)
    db.add(Message(
        session_id=session_uuid,
        role="user",
        content=user_message,
        is_Important=is_important,
    ))
    await db.flush()

    turn_start_index = len(state.get("messages", []))

    return {
        "is_user_msg_important": is_important,
        "messages": [HumanMessage(content=user_message)],
        "turn_start_index": turn_start_index,
    }
