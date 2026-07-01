# app/graph/shared/base_state.py
from typing import Annotated, List, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_dicts(left: dict, right: dict) -> dict:
    if right.get("__reset__"):
        return {}
    return {**left, **right}


class BaseAgentState(TypedDict, total=False):
    # --- SHARED CORE (Auto-syncs between parent and subgraphs) ---
    messages: Annotated[List[BaseMessage], add_messages]
    session_id: str
    user_id: str
    user_message: str
    #context memory
    summary:str
    context_window :list

    # --- SHARED RESULTS (Where all agents dump their final output) ---
    workers_needed: List[str]
    delegated_tasks: Dict[str, str]
    final_response: str
    worker_results: Annotated[Dict[str, Any], merge_dicts]
    direct_response: str

    # --- CONTROL FLOW ---
    routing_round: int
    is_final: bool
