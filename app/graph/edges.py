# app/graph/edges.py
"""
Conditional edge function: should_continue

Replaces:
    if not ai_message.tool_calls:
        ai_response = ai_message.content
        break
    ...
    iteration += 1
    # (loop condition: iteration < MAX_ITERATIONS)

Returns a string key that `add_conditional_edges` maps to the next node name.
"""

from app.graph.state import AgentGraphState

MAX_ITERATIONS = 5


def should_continue(state: AgentGraphState) -> str:
    last_message = state["messages"][-1]

    # No tool calls -> agent is done talking, proceed to UI generation.
    if not getattr(last_message, "tool_calls", None):
        return "generate_ui"

    # Safety valve: stop looping even if the model keeps requesting tools.
    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return "generate_ui"

    return "tools"
