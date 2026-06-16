# app/graph/agent_graph.py
"""
Graph assembly.

WHY A FACTORY FUNCTION (build_graph) INSTEAD OF A MODULE-LEVEL GRAPH?
-----------------------------------------------------------------------
Several nodes (load_context, classify_importance, route_tools,
persist_messages) need access to:
  - `db`: an AsyncSession (request-scoped, created per FastAPI request)
  - `user_message`: the raw string from the incoming request

These are NOT part of `AgentGraphState` (state must be checkpointer-safe /
serializable, and a DB session is neither serializable nor safe to share
across requests/threads).

LangGraph nodes normally take only `(state)` or `(state, config)`. To inject
extra per-request dependencies, we build the graph PER REQUEST using
`functools.partial` closures that bind `db` and `user_message` to the node
functions, then compile. The graph STRUCTURE (nodes/edges) is identical every
time; only the bound dependencies differ. Compilation is cheap relative to
the LLM calls, so this has negligible overhead.

The checkpointer is the one thing that SHOULD be created once (it manages its
own connection pool) and passed in.
"""

import functools
from langgraph.graph import StateGraph, START, END

from app.graph.state import AgentGraphState
from app.graph.edges import should_continue
from app.graph.nodes.load_context import load_context_node
from app.graph.nodes.classify_importance import classify_importance_node
from app.graph.nodes.inject_permanent_facts import inject_permanent_facts_node
from app.graph.nodes.route_tools import route_tools_node
from app.graph.nodes.agent import agent_node
from app.graph.nodes.tools import tools_node
from app.graph.nodes.generate_ui import generate_ui_node
from app.graph.nodes.persist_messages import persist_messages_node


def build_graph(db, user_message: str, checkpointer):
    """
    Builds and compiles the agent graph for a single request.

    Args:
        db: AsyncSession for this request (used by load_context,
            classify_importance, persist_messages).
        user_message: the raw user text for this turn (used by
            load_context, classify_importance, route_tools).
        checkpointer: a shared AsyncPostgresSaver instance (created once
            at app startup, see main.py).
    """
    graph = StateGraph(AgentGraphState)

    # Bind request-scoped dependencies via partial application.
    graph.add_node("load_context", functools.partial(load_context_node, db=db, user_message=user_message))
    graph.add_node("classify_importance", functools.partial(classify_importance_node, db=db, user_message=user_message))
    graph.add_node("inject_permanent_facts", inject_permanent_facts_node)
    graph.add_node("route_tools", functools.partial(route_tools_node, user_message=user_message))
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("generate_ui", generate_ui_node)
    graph.add_node("persist_messages", functools.partial(persist_messages_node, db=db))

    graph.add_edge(START, "load_context")
    graph.add_edge("load_context", "classify_importance")
    graph.add_edge("classify_importance", "inject_permanent_facts")
    graph.add_edge("inject_permanent_facts", "route_tools")
    graph.add_edge("route_tools", "agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "generate_ui": "generate_ui",
        },
    )

    # Loop: after tools, go back to inject_permanent_facts so a
    # change_behaviour_profile call (handled in tools_node) rebuilds the
    # system prompt with the new behaviour mode before the next agent call.
    # inject_permanent_facts -> route_tools -> agent, same as the initial path.
    graph.add_edge("tools", "inject_permanent_facts")

    graph.add_edge("generate_ui", "persist_messages")
    graph.add_edge("persist_messages", END)

    return graph.compile(checkpointer=checkpointer)
