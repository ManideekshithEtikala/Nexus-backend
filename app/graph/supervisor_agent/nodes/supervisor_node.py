# app/graph/supervisor/nodes/supervisor_node.py
import os
from pydantic import BaseModel, Field, SecretStr
from typing import List, Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.graph.supervisor_agent.state import BaseAgentState

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set")

api_key = SecretStr(groq_api_key)

supervisor_llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.0,
    api_key=api_key,
)


class RoutingDecision(BaseModel):
    # 🎯 UPDATED: Added "normal" to the allowed routing workers
    workers: List[Literal["research", "coding", "normal"]] = Field(
        description="Which worker agent(s) should handle this sequence next."
    )
    tasks: dict = Field(
        description='Task string per worker, e.g. {"research": "Search for latest revenue figures", "normal": "Have a friendly conversation about Singapore"}'
    )
    is_final: bool = Field(
        description="True if no more delegation is needed and we should respond now."
    )


# 🎯 UPDATED: Refined system rules for high-precision routing boundaries
_SUPERVISOR_SYSTEM_PROMPT = """
You are a routing controller for a multi-agent system. Your job is to classify the user's message and route it to the exact correct specialist worker agent.

Available workers:
- "normal": Handles general conversation, casual queries, opinion-based questions, small talk, greetings, or casual chat questions that do not require live lookups (e.g., "hi", "how is the day in Singapore?", "tell me a joke", "thank you").
- "research": Handles live queries requiring factual data retrieval, deep background research, pulling database metrics, or real-time web monitoring information. 
- "coding": Handles math compilation, script building, data processing, or algorithmic execution tasks.

Rules:
1. Use "normal" for any chitchat, friendly questions, banter, or casual queries that can be answered through a normal assistant chat response without deep research or tools.
2. Use "research" ONLY when the user genuinely needs deep information gathering, data synthesis, or complex lookups.
3. Use "coding" when code or algorithmic execution is required.
4. Set is_final=false whenever you route to a worker like "research", "coding", or "normal". Let the workers do the work.
5. Set is_final=true ONLY if you are absolutely sure the entire task has been completed through previous rounds and you are ready to wrap up.
6. Never invent a worker name that isn't in the list above.
7.Always give a clear response in the json format specified, and do not include any extra text outside the JSON.
8.Your output should match the exact json format of {"workers": [...], "tasks": {...}, "is_final": true/false} with no extra commentary or explanation.
"""


async def supervisor_node(state: BaseAgentState) -> dict:
    # Check for the normal agent response if there is a direct_response then is_Final =True
    existing_direct = state.get("direct_response", "")
    if existing_direct:
        print("🧭 [SUPERVISOR] Normal agent has already responded. Wrapping up.")
        return {
            "workers_needed": [],
            "delegated_tasks": {},
            "is_final": True,
            "direct_response": existing_direct,
        }

    # 🎯 OPTIMIZATION: Use the token-optimized context window for the
    # supervisor call!

    optimized_history = state.get("context_window", [])
    user_message = state.get("user_message", "")
    existing_results = state.get("worker_results", {})
    routing_round = state.get("routing_round", 0)

    print(
        f"🧭 [SUPERVISOR] incoming -> user_message={user_message!r}, "
        f"routing_round={routing_round!r}, existing_results={existing_results!r}"
    )

    results_summary = (
        "\n".join(
            f"- {worker}: {answer}" for worker, answer in existing_results.items()
        )
        if existing_results
        else "(none yet)"
    )

    routing_prompt = f"""
User message: "{user_message}"

Results gathered so far this turn:
{results_summary}

Routing round: {routing_round}

Decide the next routing step.
""".strip()

    router = supervisor_llm.with_structured_output(RoutingDecision,method="json_mode")

    try:
        # Pass the system prompt along with our filtered context window payload
        llm_payload = (
            [SystemMessage(content=_SUPERVISOR_SYSTEM_PROMPT)]
            + optimized_history
            + [HumanMessage(content=routing_prompt)]
        )

        decision = await router.ainvoke(llm_payload)

        print(
            f"🧭 [SUPERVISOR] raw decision -> workers={decision.workers!r}, "
            f"is_final={decision.is_final!r}, "
            f"tasks={decision.tasks!r}"
        )

    except Exception as e:
        print(f"⚠️ [SUPERVISOR] Routing decision failed: {e}")
        # Safeguard fallback
        decision = RoutingDecision(
            workers=["normal"], tasks={"normal": user_message}, is_final=True
        )

    output = {
        "workers_needed": decision.workers,
        "delegated_tasks": decision.tasks,
        "is_final": decision.is_final,
        # We clear direct_response since our new normal_agent_node will fill it out cleanly
        "direct_response":"",
        "routing_round": routing_round + 1,
    }
    print(f"🧭 [SUPERVISOR] returning state patch -> {output!r}")
    return output
