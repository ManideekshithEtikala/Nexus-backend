# app/graph/nodes/route_tools.py
"""
Stage 3 node: route_tools

Wraps your existing `semantic_tool_router`. Runs immediately before `agent`
on every iteration of the ReAct loop (selects fresh tools each time, exactly
like your original `while` loop did).

We store the selected tool NAMES in state (`active_tool_names`) rather than
the tool objects themselves, since state should stay JSON-serializable for
the checkpointer. `agent_node` re-resolves names -> tool objects from
TOOL_REGISTRY.
"""

from app.services.router import semantic_tool_router
from app.tools.state_immutability import TOOL_REGISTRY
from app.graph.state import AgentGraphState


async def route_tools_node(state: AgentGraphState, user_message: str) -> dict:
    from app.core.schema import BehaviourPattern
    current_behaviour_value = state.get("current_behaviour", BehaviourPattern.CASUAL_PRODUCTIVITY.value)
    active_tools = await semantic_tool_router(user_message, current_behaviour_value)
    active_tool_names = [t.name for t in active_tools if t.name in TOOL_REGISTRY]

    return {"active_tool_names": active_tool_names}
