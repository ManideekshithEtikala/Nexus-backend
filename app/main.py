import os
import uuid
import asyncio
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import SecretStr

# Database & LangChain
from sqlalchemy.ext.asyncio import AsyncSession
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from sqlalchemy import text
# 🏗️ ARCHITECTURE SERVICES (Our clean, isolated files!)
from .database import get_db
from .database.database_neo4j import neo4j_client
from .models.model import Message

from .tools.state_immutability import TOOL_REGISTRY, safe_execute_tool
from .core.schema import AgentState, UserMessage, ResumeAction, BehaviourPattern, KnowledgeGraphUpdate

# 🚀 NEW: Import our isolated Memory logic
from app.services.memory import get_or_create_session, populate_state_context, compress_old_history_background, classifier_chain
from app.services.router import semantic_tool_router
from app.services.extractor import generate_ui_pipeline

# =====================================================================
# ⚙️ SYSTEM INITIALIZATION
# =====================================================================
load_dotenv()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000","https://nexus-frontned-9xnj8qgfy-manideekshithetikalas-projects.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.environ["GROQ_API_KEY"]
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.1, api_key=SecretStr(api_key))

MAX_ITERATIONS = 5
REQUIRES_APPROVAL_TOOLS = ["send_email"]


@app.get("/api/health")
async def database_check(db: AsyncSession = Depends(get_db)):
    """Validates connectivity to both PostgreSQL (Supabase) and Neo4j."""
    try:
        # 1. Test PostgreSQL
        await db.execute(text("SELECT 1"))
        
        # 2. Test Neo4j
        await neo4j_client.test_connection()
        
        return {"status": "healthy", "postgres": "connected", "neo4j": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
# =====================================================================
# 🛣️ ROUTE 1: PRIMARY AGENT ENDPOINT
# =====================================================================
@app.post("/api/agent")
async def user_message(payload: UserMessage, db: AsyncSession = Depends(get_db)):
    """
    The main gateway. Receives user text, loads memory, runs the ReAct loop, 
    and returns a clean UI JSON Pipeline.
    """
    # 1. Rehydrate State from PostgreSQL
    session_uuid = uuid.UUID(payload.sessionId)
    chat_session = await get_or_create_session(db, str(session_uuid), payload.message)
    state = AgentState(session_id=payload.sessionId, current_summary=chat_session.summary or "")
    db_messages = await populate_state_context(db, state)

    # 2. Tag User Message Importance & Save to DB
    try:
        user_classification = await classifier_chain.ainvoke({"content": payload.message})
        state.is_user_msg_important = user_classification.is_Important
    except Exception:
        state.is_user_msg_important = False
        
    state.messages.append(HumanMessage(content=payload.message))
    db.add(Message(session_id=session_uuid, role="user", content=payload.message, is_Important=state.is_user_msg_important))

    # 3. Background Graph Memory Extraction (Fire & Forget)
    async def extract_and_save_graph(user_text: str):
        try:
            extractor_llm_graph = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=SecretStr(api_key))
            graph_extractor = extractor_llm_graph.with_structured_output(KnowledgeGraphUpdate)
            graph_update = await graph_extractor.ainvoke(f"Extract core facts as nodes/edges: '{user_text}'")
            await neo4j_client.execute_graph_update(graph_update.model_dump())
        except Exception as e:
            print(f"⚠️ Graph extraction failed: {str(e)}")

    asyncio.create_task(extract_and_save_graph(payload.message))

    # 4. Inject Permanent Truth from Neo4j
    permanent_facts_text = await neo4j_client.fetch_user_graph_facts()

    # 5. INITIALIZE THE REACT LOOP
    BEHAVIOUR_MAP = {
        BehaviourPattern.CASUAL_PRODUCTIVITY: """
        You are Nexus, an advanced execution agent.

=== HIERARCHY OF TRUTH ===
1. PERMANENT GRAPH FACTS: {permanent_memory}
2. RECENT CHAT SUMMARY: {summary}

INSTRUCTIONS:
- The Graph Facts represent absolute truth. If the Recent Chat Summary contradicts the Graph Facts, ALWAYS trust the Graph Facts.
- You are connected to a live execution environment. NEVER say "As an AI, I cannot do this." ALWAYS use the tools provided to you.
- Output clean, professional Markdown. Do not use filler words like "Here is the table you asked for." Just output the table.""",
        BehaviourPattern.DEEP_RESEARCH: """
        You are Nexus, an advanced DeepSearch agent.

=== HIERARCHY OF TRUTH ===
1. PERMANENT GRAPH FACTS: {permanent_memory}
2. RECENT CHAT SUMMARY: {summary}

INSTRUCTIONS:
- The Graph Facts represent absolute truth. If the Recent Chat Summary contradicts the Graph Facts, ALWAYS trust the Graph Facts.
- You are connected to a live execution environment. NEVER say "As an AI, I cannot do this." ALWAYS use the tools provided to you.
- Output clean, professional Markdown. Do not use filler words like "Here is the table you asked for." Just output the table.""",
        BehaviourPattern.STANDARD_CODING: """
        You are Nexus, an advanced Coding agent.

=== HIERARCHY OF TRUTH ===
1. PERMANENT GRAPH FACTS: {permanent_memory}
2. RECENT CHAT SUMMARY: {summary}

INSTRUCTIONS:
- The Graph Facts represent absolute truth. If the Recent Chat Summary contradicts the Graph Facts, ALWAYS trust the Graph Facts.
- You are connected to a live execution environment. NEVER say "As an AI, I cannot do this." ALWAYS use the tools provided to you.
- Output clean, professional Markdown. Do not use filler words like "Here is the table you asked for." Just output the table."""
    }

    initial_prompt = BEHAVIOUR_MAP.get(state.current_behaviour).format(summary=state.current_summary, permanent_memory=permanent_facts_text)
    
    if state.messages and isinstance(state.messages[0], SystemMessage):
        state.messages[0] = SystemMessage(content=initial_prompt)
    else:
        state.messages.insert(0, SystemMessage(content=initial_prompt))

    input_length = len(state.messages)
    iteration = 0
    ai_response = ""
    
    # 🔄 THE COMPUTATIONAL LOOP
    while iteration < MAX_ITERATIONS:
        # Route tools dynamically based on the active conversation
        active_tools = await semantic_tool_router(payload.message, state.current_behaviour.value)
        current_llm = llm.bind_tools(active_tools) if active_tools else llm

        ai_message = await current_llm.ainvoke(state.messages)
        state.messages.append(ai_message)

        # Break loop if the AI is just talking (no tools)
        if not ai_message.tool_calls:
            ai_response = ai_message.content
            break

        primary_tool_call = ai_message.tool_calls[0]
        tool_name = primary_tool_call["name"]
        tool_args = primary_tool_call["args"]
        tool_call_id = primary_tool_call["id"]

        # 🛑 HUMAN-IN-THE-LOOP CIRCUIT BREAKER
        if tool_name in REQUIRES_APPROVAL_TOOLS:
            print(f"🛑 [HIBERNATION] Pausing execution for {tool_name}")
            return {
                "status": "requires_approval",
                "session_id": payload.sessionId,
                "pending_action": {"tool_name": tool_name, "tool_args": tool_args, "tool_call_id": tool_call_id},
                "message": f"The agent wants to execute {tool_name}. Do you approve?"
            }

        # Execute safe tools and feed the result back to the LLM
        if tool_name in TOOL_REGISTRY:
            tool_msg = await safe_execute_tool(TOOL_REGISTRY[tool_name], tool_args, tool_call_id)
            state.messages.append(tool_msg)
        iteration += 1

    if not ai_response:
        for msg in reversed(state.messages[input_length:]):
            if isinstance(msg, AIMessage) and msg.content:
                ai_response = msg.content
                break

    # 6. PYDANTIC FIREWALL: EXTRACT UI JSON
    final_structured_payload = await generate_ui_pipeline(state.messages[input_length:], ai_response)

    # 7. COMMIT NEW TRANSACTIONS TO DATABASE
    state.new_agent_messages = state.messages[input_length:]
    for msg in state.new_agent_messages:
        content_payload = str(msg.content) if msg.content else ""
        if isinstance(msg, AIMessage):
            t_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None
            db.add(Message(session_id=session_uuid, role="assistant", content=content_payload, tool_calls=t_calls, is_Important=False))
        elif isinstance(msg, ToolMessage):
            t_calls = {"id": getattr(msg, "tool_call_id", "legacy")}
            db.add(Message(session_id=session_uuid, role="tool", content=content_payload, tool_calls=t_calls, is_Important=False))

    await db.flush()
    await compress_old_history_background(chat_session, db_messages)
    await db.commit()

    return {
        "status": "success",
        "session_id": payload.sessionId,
        "data": final_structured_payload.model_dump(),
    }


# =====================================================================
# 🛣️ ROUTE 2: RESUME HIBERNATED AGENT
# =====================================================================
@app.post("/api/agent/resume")
async def resume_agent(payload: ResumeAction, db: AsyncSession = Depends(get_db)):
    """Wakes the agent up after the React frontend sends a Human Approval."""
    session_uuid = uuid.UUID(payload.sessionId)
    chat_session = await get_or_create_session(db, str(session_uuid), "Resumed Session")

    state = AgentState(session_id=payload.sessionId, current_summary=chat_session.summary or "")
    db_messages = await populate_state_context(db, state)
    input_length = len(state.messages)

    # 1. INJECT HUMAN DECISION
    if payload.is_approved:
        tool_message = await safe_execute_tool(TOOL_REGISTRY[payload.tool_name], payload.tool_args, payload.tool_call_id)
    else:
        tool_message = ToolMessage(
            content=f"ERROR: The human explicitly REJECTED this action.",
            tool_call_id=payload.tool_call_id,
            status="error"
        )
    state.messages.append(tool_message)

    # 2. RESUME REACT LOOP
    iteration = 0
    ai_response = ""
    while iteration < MAX_ITERATIONS:
        active_tools = await semantic_tool_router("User responded to approval request.", state.current_behaviour.value)
        current_llm = llm.bind_tools(active_tools) if active_tools else llm
        
        ai_message = await current_llm.ainvoke(state.messages)
        state.messages.append(ai_message)

        if not ai_message.tool_calls:
            ai_response = ai_message.content
            break

        primary_tool_call = ai_message.tool_calls[0]
        if primary_tool_call["name"] in TOOL_REGISTRY:
            tool_msg = await safe_execute_tool(TOOL_REGISTRY[primary_tool_call["name"]], primary_tool_call["args"], primary_tool_call["id"])
            state.messages.append(tool_msg)
        iteration += 1

    # 3. PYDANTIC FIREWALL
    final_structured_payload = await generate_ui_pipeline(state.messages[input_length:], ai_response)

    # 4. SAVE TO DB
    for msg in state.messages[input_length:]:
        if isinstance(msg, AIMessage):
            db.add(Message(session_id=session_uuid, role="assistant", content=str(msg.content), is_Important=False))
        elif isinstance(msg, ToolMessage):
            t_calls = {"id": getattr(msg, "tool_call_id", "legacy")}
            db.add(Message(session_id=session_uuid, role="tool", content=str(msg.content), tool_calls=t_calls, is_Important=False))
            
    await db.commit()

    return {
        "status": "success",
        "session_id": payload.sessionId,
        "data": final_structured_payload.model_dump(),
    }