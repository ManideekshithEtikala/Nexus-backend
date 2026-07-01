# app/graph/research_agent/research_graph.py
from langgraph.graph import StateGraph, START, END
from app.graph.supervisor_agent.state import BaseAgentState
from app.graph.research_agent.nodes.research_node import (
    load_research_context_node,
    call_research_llm_node,
    export_research_results_node,
)
from app.graph.research_agent.edge import should_execute_research


def build_research_graph() -> StateGraph:
    # Uses the exact same BaseAgentState type contract
    sub_graph = StateGraph(BaseAgentState)

    
    # Register Nodes
    sub_graph.add_node("load_context", load_research_context_node)
    sub_graph.add_node("call_llm", call_research_llm_node)
    sub_graph.add_node("export_results", export_research_results_node)

    # Define Local Flow
    sub_graph.add_edge(START, "load_context")

    # Conditional gate: If there's no task, skip directly to ending
    sub_graph.add_conditional_edges(
        "load_context", should_execute_research, {"continue": "call_llm", "skip": END}
    )

    sub_graph.add_edge("call_llm", "export_results")
    sub_graph.add_edge("export_results", END)

    # Compile sub-graph
    return sub_graph.compile()
