# app/graph/state.py
"""
LangGraph state definition.

This replaces the Pydantic `AgentState` for graph execution purposes.
LangGraph requires the state to be a TypedDict (or dataclass) so it can apply
reducers (like `add_messages`) to individual fields and persist/merge partial
updates returned by each node.

NOTE: Your existing Pydantic `AgentState` in app/core/schema.py is NOT removed.
It can still be used anywhere else in your codebase (e.g. if you build
non-graph helper functions that want a validated object). For the graph itself,
this TypedDict is the source of truth.
"""

from typing import Annotated, List, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentGraphState(TypedDict, total=False):
    # --- Core conversation ---
    # `add_messages` reducer: new messages returned by a node are APPENDED
    # to this list automatically (not overwritten). This is the LangGraph
    # equivalent of `state.messages.append(...)`.
    messages: Annotated[List[BaseMessage], add_messages]

    # --- Session / identity ---
    session_id: str

    # --- Memory / summary ---
    current_summary: Optional[str]
    permanent_memory: str

    # --- Behaviour routing ---
    # Stored as the plain string VALUE of BehaviourPattern (not the Enum
    # object itself). The checkpointer serializes state to JSON/msgpack;
    # custom Enum classes need extra registration to round-trip safely.
    # Storing `.value` (a plain str, since BehaviourPattern is a str-Enum)
    # avoids that entirely. Convert with BehaviourPattern(value) wherever
    # the Enum type is needed (e.g. BEHAVIOUR_MAP lookups).
    current_behaviour: str

    # --- Importance tagging ---
    is_user_msg_important: bool

    # --- Loop control ---
    iteration: int

    # --- Tool routing ---
    active_tool_names: List[str]

    # --- Bookkeeping for persistence ---
    # Index into `messages` marking where THIS turn's new messages begin.
    # Used by persist_messages to know what to write to Postgres.
    turn_start_index: int

    # --- Final structured output ---
    ui_pipeline: Optional[dict]
