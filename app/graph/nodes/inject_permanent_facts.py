# app/graph/nodes/inject_permanent_facts.py
"""
Stage 5 node: inject_permanent_facts

From your original main.py:
  - permanent_facts_text = await neo4j_client.fetch_user_graph_facts()
  - BASE_BEHAVIOUR_TEMPLATE / BEHAVIOUR_MAP construction
  - inserting/replacing the SystemMessage at messages[0]

This node:
  1. Fetches permanent facts from Neo4j.
  2. Builds the behaviour-specific system prompt using BEHAVIOUR_MAP.
  3. Updates the SystemMessage at the front of `messages`.

POSITION-PRESERVING REPLACE:
`load_context_node` guarantees a placeholder SystemMessage with id
SYSTEM_MESSAGE_ID already exists at index 0 of `messages` (either freshly
seeded on turn 1, or restored by the checkpointer on later turns). The
`add_messages` reducer replaces a message IN PLACE (same position) when the
returned message has a matching `id` — it does NOT move it. So we simply
return `{"messages": [system_message]}` with the same id, and its content
gets updated at index 0 without disturbing message order.
"""

from langchain_core.messages import SystemMessage

from app.database.database_neo4j import neo4j_client
from app.graph.state import AgentGraphState
from app.core.schema import BehaviourPattern
from app.graph.nodes.load_context import SYSTEM_MESSAGE_ID

BASE_BEHAVIOUR_TEMPLATE = """
You are Nexus, an advanced execution agent.

## Hierarchy of truth
1. Permanent graph facts: {permanent_memory}
2. Recent chat summary: {summary}

## Global rules
- Permanent graph facts are the source of truth.
- If the recent chat summary conflicts with permanent graph facts, trust permanent graph facts.
- Use the available tools whenever they are relevant to complete the task.
- Never claim lack of capability if a provided tool can perform the action.
- Be concise, accurate, and action-oriented.
- Output clean professional Markdown.
- Avoid filler such as "Here is..." or "Below is...".
"""

BEHAVIOUR_MAP = {
    BehaviourPattern.CASUAL_PRODUCTIVITY: BASE_BEHAVIOUR_TEMPLATE + """
## Behaviour mode: Casual Productivity
- Prioritize speed, clarity, and practical execution.
- Give direct answers first.
- Use lightweight reasoning unless the task clearly needs deeper analysis.
- Prefer short plans, checklists, summaries, drafts, and actionable next steps.
- Ask at most one clarifying question only when the task is blocked.
- Keep the response compact and useful for fast iteration.
""",

    BehaviourPattern.DEEP_RESEARCH: BASE_BEHAVIOUR_TEMPLATE + """
## Behaviour mode: Deep Research
- Prioritize completeness, verification, and nuance.
- Break the problem into sub-questions before answering.
- Cross-check important claims against available evidence and tools.
- Call out uncertainty, assumptions, trade-offs, and conflicting signals explicitly.
- Structure responses with clear sections, comparisons, and synthesized findings.
- Prefer depth over brevity when the task benefits from it.
""",

    BehaviourPattern.STANDARD_CODING: BASE_BEHAVIOUR_TEMPLATE + """
## Behaviour mode: Standard Coding
- Prioritize correctness, maintainability, and working solutions.
- Produce code that is directly usable and consistent with the user's stack.
- State assumptions briefly when requirements are missing, then proceed.
- When coding, include only necessary explanation; let the code stay central.
- Prefer simple, robust implementations over clever ones.
- When debugging, identify root cause first, then propose the fix.
- Preserve existing interfaces and behavior unless change is requested.
""",

    BehaviourPattern.CRITICAL_REFLECTIVE: BASE_BEHAVIOUR_TEMPLATE + """
## Behaviour mode: Critical Reflective
- Prioritize careful evaluation, weighing pros/cons, and identifying risks.
- Question assumptions (including the user's) where warranted.
- Provide balanced perspectives before recommending a course of action.
""",
}


async def inject_permanent_facts_node(state: AgentGraphState) -> dict:
    permanent_facts_text = await neo4j_client.fetch_user_graph_facts()

    behaviour_value = state.get("current_behaviour") or BehaviourPattern.CASUAL_PRODUCTIVITY.value
    try:
        behaviour = BehaviourPattern(behaviour_value)
    except ValueError:
        behaviour = BehaviourPattern.CASUAL_PRODUCTIVITY

    template = BEHAVIOUR_MAP.get(behaviour, BEHAVIOUR_MAP[BehaviourPattern.CASUAL_PRODUCTIVITY])

    prompt_text = template.format(
        summary=state.get("current_summary", "") or "",
        permanent_memory=permanent_facts_text,
    )

    system_message = SystemMessage(content=prompt_text, id=SYSTEM_MESSAGE_ID)

    return {
        "permanent_memory": permanent_facts_text,
        # Replaces the existing SystemMessage in place (same id, same index).
        "messages": [system_message],
    }
