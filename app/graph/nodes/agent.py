# app/graph/nodes/agent.py
"""
Stage 2 node: agent

The core LLM call. Equivalent to:
    current_llm = llm.bind_tools(active_tools) if active_tools else llm
    ai_message = await current_llm.ainvoke(state.messages)
    state.messages.append(ai_message)

Tool objects are re-resolved from `active_tool_names` (set by route_tools_node)
via TOOL_REGISTRY, since state stores names (JSON-serializable) not objects.

This node also increments `iteration`, which `should_continue` uses to
enforce MAX_ITERATIONS (replacing your `while iteration < MAX_ITERATIONS`).
"""

import os
from pydantic import SecretStr
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage

from app.tools.state_immutability import TOOL_REGISTRY
from app.graph.state import AgentGraphState

api_key = os.environ["GROQ_API_KEY"]
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    api_key=SecretStr(api_key),
)


async def agent_node(state: AgentGraphState) -> dict:
    active_tool_names = state.get("active_tool_names", [])
    active_tools = [
        TOOL_REGISTRY[name] for name in active_tool_names if name in TOOL_REGISTRY
    ]

    current_llm = llm.bind_tools(active_tools) if active_tools else llm

    try:
        ai_message = await current_llm.ainvoke(state["messages"])
    except Exception as e:
        # Groq's tool-calling sometimes fails with 400 tool_use_failed when
        # the model produces malformed function-call syntax (common with
        # llama-4-scout on code-heavy or ambiguous-tool-choice prompts).
        # RETRY ONCE without tools bound, so the model can at least answer
        # in plain text instead of crashing the whole graph.
        print(f"⚠️ [AGENT] LLM call with tools failed: {e}. Retrying without tools.")
        try:
            ai_message = await llm.ainvoke(state["messages"])
        except Exception as e2:
            print(f"⚠️ [AGENT] Retry without tools also failed: {e2}")
            ai_message = AIMessage(
                content="I ran into an error trying to process that request. Could you rephrase it?"
            )

    return {
        "messages": [ai_message],
        "iteration": state.get("iteration", 0) + 1,
    }
