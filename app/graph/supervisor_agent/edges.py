# app/graph/supervisor/edges.py
from app.graph.supervisor_agent.state import BaseAgentState

MAX_ROUTING_ROUNDS = 4


def route_to_worker(state: BaseAgentState) -> str:
    is_final = state.get("is_final", False)
    workers_needed = state.get("workers_needed", [])
    routing_round = state.get("routing_round", 0)

    print(
        f"🚦 [GATE:route_to_worker] is_final={is_final!r}, "
        f"workers_needed={workers_needed!r}, routing_round={routing_round!r}"
    )

    if routing_round >= MAX_ROUTING_ROUNDS:
        print("🚦 [GATE:route_to_worker] -> finalize (max rounds hit)")
        return "finalize"

    if is_final or not workers_needed:
        print("🚦 [GATE:route_to_worker] -> finalize (is_final or no workers)")
        return "finalize"

    next_worker = workers_needed[0]

    if next_worker == "research":
        print("🚦 [GATE:route_to_worker] -> research_agent")
        return "research_agent"
    if next_worker == "normal":
        print("🚦 [GATE:route_to_worker] -> normal_agent")
        return "normal_agent"
    if next_worker == "coding":
        print("🚦 [GATE:route_to_worker] -> coding_agent")
        return "coding_agent"

    print(
        f"🚦 [GATE:route_to_worker] -> finalize (unrecognized worker {next_worker!r})"
    )
    return "finalize"


def should_continue_routing(state: BaseAgentState) -> str:
    routing_round = state.get("routing_round", 0)
    worker_results = state.get("worker_results", {})
    if "normal" in worker_results and worker_results["normal"]:
        print("🚦 [EDGE] Chat answer detected from normal agent. Going to finalize.")
        return "finalize"
    if routing_round >= MAX_ROUTING_ROUNDS:
        print(f"🚦 [GATE:should_continue_routing] round={routing_round} -> finalize")
        return "finalize"

    print(f"🚦 [GATE:should_continue_routing] round={routing_round} -> compress_memory")
    return "compress_memory"
