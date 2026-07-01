
from langgraph.graph import StateGraph, START, END
from app.graph.supervisor_agent.state import BaseAgentState
from app.graph.normal_agent.nodes.normal_agent import normal_agent_node

def build_normal_agent_graph()->StateGraph:
    """
    Builds an ultra-simple, non-complex sequential sub-graph.
    Flow: START -> normal_agent_node -> END
    """
    sub_graph = StateGraph(BaseAgentState)

    # Register the single node
    sub_graph.add_node("normal_agent_node", normal_agent_node)

    # Simple linear flow completely free of conditional gates
    sub_graph.add_edge(START, "normal_agent_node")
    sub_graph.add_edge("normal_agent_node", END)

    return sub_graph.compile()
