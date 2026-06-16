# app/graph/nodes/generate_ui.py
"""
Stage 6 node: generate_ui

Wraps `generate_ui_pipeline` (extractor.py) unchanged. Takes this turn's new
messages (everything after `turn_start_index`) plus a fallback text, and
forces them through the structured-output extractor.

`ai_response` fallback: same logic as your original code — walk backwards
through this turn's messages to find the last non-empty AIMessage content.
"""

from langchain_core.messages import AIMessage

from app.services.extractor import generate_ui_pipeline
from app.graph.state import AgentGraphState


async def generate_ui_node(state: AgentGraphState) -> dict:
    turn_start_index = state.get("turn_start_index", 0)
    new_messages = state["messages"][turn_start_index:]

    ai_response = ""
    for msg in reversed(new_messages):
        if isinstance(msg, AIMessage) and msg.content:
            ai_response = msg.content
            break

    final_structured_payload = await generate_ui_pipeline(new_messages, ai_response)

    return {"ui_pipeline": final_structured_payload.model_dump()}
