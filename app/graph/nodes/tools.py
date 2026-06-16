# app/graph/nodes/tools.py
"""
Stage 2 + 7 node: tools

Combines:
  - REQUIRES_APPROVAL_TOOLS check (now via interrupt())
  - safe_execute_tool (your validation/execution firewall, unchanged)

HOW interrupt() WORKS HERE:
  When the agent calls a tool in REQUIRES_APPROVAL_TOOLS (e.g. "send_email"),
  this node calls `interrupt({...payload...})`. This raises a special
  exception internally that LangGraph catches: it SAVES the current graph
  state via the checkpointer and STOPS execution, returning an
  `__interrupt__` entry in the result of `graph.ainvoke(...)`.

  Your FastAPI route inspects that and returns the "requires_approval"
  response to the frontend (same shape as your original early-return).

  To resume: the frontend calls /api/agent/resume, and your route calls:
      graph.ainvoke(Command(resume={"is_approved": ..., ...}), config=...)

  Execution resumes INSIDE this node, with `interrupt()` now returning the
  value passed to `Command(resume=...)`. We use that to either execute the
  tool (if approved) or build an error ToolMessage (if rejected) — exactly
  matching your original `/api/agent/resume` logic.

IMPORTANT: `interrupt()` re-runs the node from the top on resume. Code BEFORE
`interrupt()` in this node will execute AGAIN. Keep everything before
`interrupt()` side-effect-free (which it already is here — just inspecting
the tool call).
"""

from langgraph.types import interrupt
from langchain_core.messages import ToolMessage

from app.tools.state_immutability import TOOL_REGISTRY, safe_execute_tool
from app.graph.state import AgentGraphState
from app.core.schema import BehaviourPattern

REQUIRES_APPROVAL_TOOLS = ["send_email"]


async def tools_node(state: AgentGraphState) -> dict:
    last_message = state["messages"][-1]
    primary_tool_call = last_message.tool_calls[0]

    tool_name = primary_tool_call["name"]
    tool_args = primary_tool_call["args"]
    tool_call_id = primary_tool_call["id"]

    if tool_name in REQUIRES_APPROVAL_TOOLS:
        # PAUSE HERE. Graph state is checkpointed; execution stops until
        # /api/agent/resume sends Command(resume=...).
        decision = interrupt({
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_call_id": tool_call_id,
            "message": f"The agent wants to execute {tool_name}. Do you approve?",
        })

        # --- Execution resumes here after Command(resume=decision) ---
        is_approved = decision.get("is_approved", False)

        if is_approved:
            tool_msg = await safe_execute_tool(TOOL_REGISTRY[tool_name], tool_args, tool_call_id)
        else:
            tool_msg = ToolMessage(
                content="ERROR: The human explicitly REJECTED this action.",
                tool_call_id=tool_call_id,
                status="error",
            )

        return {"messages": [tool_msg]}

    # Normal (non-approval) tool execution
    if tool_name in TOOL_REGISTRY:
        tool_msg = await safe_execute_tool(TOOL_REGISTRY[tool_name], tool_args, tool_call_id)

        update: dict = {"messages": [tool_msg]}

        # IMPROVEMENT OVER ORIGINAL: actually apply the behaviour change to
        # graph state. In the original code, `change_behaviour_profile`
        # returned a confirmation string but `state.current_behaviour` was
        # never updated, so BEHAVIOUR_MAP always used the initial default
        # for the whole request. Here, a successful call updates state, and
        # inject_permanent_facts will rebuild the system prompt on the next
        # loop iteration (route_tools -> agent) with the new behaviour.
        if tool_name == "change_behaviour_profile" and tool_msg.status == "success":
            new_profile = tool_args.get("new_profile")
            if new_profile in BehaviourPattern._value2member_map_:
                update["current_behaviour"] = new_profile

        return update

    # Tool name not found in registry — return an error ToolMessage so the
    # agent loop can recover instead of crashing.
    return {
        "messages": [
            ToolMessage(
                content=f"Error: Tool '{tool_name}' is not registered.",
                tool_call_id=tool_call_id,
                status="error",
            )
        ]
    }
