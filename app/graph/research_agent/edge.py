# app/graph/research_agent/edges.py
from app.graph.supervisor_agent.state import BaseAgentState


def should_execute_research(state: BaseAgentState) -> str:
    """Determines whether the research agent should execute or pass."""
    active_task = state.get("user_message", "")
    decision = (
        "skip" if (not active_task or len(active_task.strip()) == 0) else "continue"
    )
    print(
        f"🚦 [GATE:should_execute_research] active_task={active_task!r} -> {decision}"
    )
    if decision == "skip":
        return "skip"
    return "continue"
