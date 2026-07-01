# app/graph/supervisor/supervisor_graph.py
from langgraph.graph import StateGraph, START, END

from app.graph.supervisor_agent.state import BaseAgentState
from app.graph.supervisor_agent.edges import route_to_worker, should_continue_routing
from app.graph.supervisor_agent.nodes.supervisor_node import supervisor_node
from app.graph.research_agent.research_graph import build_research_graph
from app.graph.normal_agent.normal_agent_graph import build_normal_agent_graph
from app.graph.supervisor_agent.nodes.compress_memory import compress_memory_node

async def _dummy_coding(state: BaseAgentState) -> dict:
    delegated_tasks = state.get("delegated_tasks", {})
    task = delegated_tasks.get("knowledge", "")
    print(f"📚 [DUMMY CODING AGENT] received task: {task}")
    return {
        "worker_results": {"coding": f"(stub) found: {task}"},
    }


async def finalize_node(state: BaseAgentState) -> dict:
    direct_response = state.get("direct_response", "")
    worker_results = state.get("worker_results", {})

    print(
        f"🏁 [FINALIZE] direct_response={direct_response!r}, "
        f"worker_results={worker_results!r}"
    )

    if direct_response:
        return {"final_response": direct_response}
    combined = "\n\n".join(str(v) for v in worker_results.values() if v)
    final = combined or "No response generated."
    print(f"🏁 [FINALIZE] -> using combined worker_results, final={final[:200]!r}")
    return {
        "final_response": final,
    }


def build_supervisor_graph(checkpointer):
    graph = StateGraph(BaseAgentState)
    graph.add_node("compress_memory", compress_memory_node)
    graph.add_node("supervisor", supervisor_node)

    # 🎯 ADDED: Mount your new modular friendly agent subgraph node
    normal_graph = build_normal_agent_graph()
    graph.add_node("normal_agent", normal_graph)

    research_graph = build_research_graph()
    graph.add_node("research_agent", research_graph)

    graph.add_node("coding_agent", _dummy_coding)
    graph.add_node("finalize", finalize_node)

    # Core starting sequence
    graph.add_edge(START, "compress_memory")
    graph.add_edge("compress_memory", "supervisor")

    # 🎯 UPDATED: Map the supervisor router choices to point to the normal agent node
    graph.add_conditional_edges(
        "supervisor",
        route_to_worker,
        {
            "normal_agent": "normal_agent", 
            "research_agent": "research_agent",
            "coding_agent": "coding_agent",
            "finalize": "finalize",
        },
    )
    graph.add_conditional_edges(
        "normal_agent",
        should_continue_routing,
        {"compress_memory": "compress_memory", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "research_agent",
        should_continue_routing,
        {"compress_memory": "compress_memory", "finalize": "finalize"},
    )
    graph.add_conditional_edges(
        "coding_agent",
        should_continue_routing,
        {"supervisor": "compress_memory", "finalize": "finalize"},
    )

    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)
