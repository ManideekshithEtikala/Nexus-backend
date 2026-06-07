from enum import Enum
import os
import uuid
import json
from datetime import datetime
from typing import List, Any, Optional, Literal, Dict
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

# LangChain Core Messaging Ecosystem
from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate

# Native Database Dependencies
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .database import get_db, engine, Base
from .models import ChatSession, Message

# Specialized Agent Tools & Base Configuration
from .prompts import SYSTEM_PROMPT

# FIXED: Correct clean registry and safe execution wrapper import path
from .tools.state_immutability import TOOL_REGISTRY, safe_execute_tool

import asyncio
from .database.database_neo4j import neo4j_client
from .models.model import KnowledgeGraphUpdate  # Or wherever you saved the schema
load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3005",
        "http://127.0.0.1:3005",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ====================================================================
# 🧠 STEP 1: DEFINE CORE PYDANTIC SCHEMAS
# =====================================================================


class BehaviourPattern(str, Enum):
    STANDARD_CODING = "STANDARD_CODING"
    DEEP_RESEARCH = "DEEP_RESEARCH"
    CASUAL_PRODUCTIVITY = "CASUAL_PRODUCTIVITY"
    CRITICAL_REFLECTIVE = "CRITICAL_REFLECTIVE"


class AgentState(BaseModel):
    session_id: str
    messages: List[BaseMessage] = Field(default_factory=list)
    current_summary: Optional[str] = ""
    is_user_msg_important: bool = False
    is_ai_msg_important: bool = False
    new_agent_messages: List[BaseMessage] = Field(default_factory=list)
    current_behaviour: BehaviourPattern = Field(
        default=BehaviourPattern.CASUAL_PRODUCTIVITY
    )

    model_config = {"arbitrary_types_allowed": True}


class UserMessage(BaseModel):
    message: str
    sessionId: str


class StandardizedRow(BaseModel):
    id: str = Field(description="The row item index or identifier (e.g. '1', '2')")
    content: str = Field(
        description="The text content, description, value, or benefit for this row."
    )


class TableColumn(BaseModel):
    id: str = Field(
        description="The unique key used for this column inside the row objects. Must match exactly the key name in rows."
    )
    name: str = Field(
        description="The human-readable column display header title (e.g. 'Category', 'AI AgentIC Engineering', 'MLOps')."
    )


class ScalableTable(BaseModel):
    columns: List[TableColumn] = Field(
        description="The structural layout columns definition."
    )
    rows: List[Dict[str, Any]] = Field(
        description=(
            "List of row objects. Crucial: Each row object MUST be a flat dictionary where keys MATCH "
            "the 'id' fields of your columns exactly. For example, if columns are ['Category', 'MLOps'], "
            "then rows must look like: {'Category': 'Pros', 'MLOps': 'Provides robust tracing and deployments'}."
        )
    )


class UIBlock(BaseModel):
    block_type: Literal["markdown_text", "citations", "data_table", "action_status"] = (
        Field(description="The target component identifier widget mapping rules.")
    )
    markdown_text: Optional[str] = Field(
        default=None, description="Use this field ONLY if block_type is 'markdown_text'"
    )
    table_data: Optional[ScalableTable] = Field(
        default=None, description="Use this field ONLY if block_type is 'data_table'"
    )


class ScalableAgentResponseSchema(BaseModel):
    ui_pipeline: List[UIBlock] = Field(
        description="An ordered serial list of UI presentation blocks to assemble the final viewport app interface stack."
    )

#pydantic model to extract the necessary tool information from the llm based on the user messgage and we use a simple llm for this so that no heavy process is there
class ToolSelection(BaseModel):
    selected_tools: List[str] = Field(
        description="A list of the tool names selected from the registry. MUST always include 'change_behaviour_profile'."
    )

class ResumeAction(BaseModel):
    sessionId: str
    tool_name: str
    tool_args: dict
    tool_call_id: str
    is_approved: bool

api_key = os.environ["GROQ_API_KEY"]
tools = list(TOOL_REGISTRY.values())
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.1,
    api_key=SecretStr(api_key),
)
MAX_ITERATIONS = 5
# Tools that can modify state, spend money, or communicate externally
REQUIRES_APPROVAL_TOOLS = ["send_email"]
# =====================================================================
# ⚙️ STEP 2: INITIALIZE THE dedicated GATEWAY EXTRACTOR CHAIN
# =====================================================================
# Temperature 0.0 forces strict adherence to our structural json definitions
extractor_llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.0, api_key=SecretStr(api_key)
)
structured_extractor = extractor_llm.with_structured_output(ScalableAgentResponseSchema)


class MemoryTaggingSchema(BaseModel):
    is_Important: bool = Field(
        description="True context if text metadata should be permanently pinned."
    )


tagging_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an elite database state memory optimization controller. Evaluate the structural value of the text.",
        ),
        (
            "human",
            "Analyze the following content and extract if it should be marked as permanently important:\n\nContent: {content}",
        ),
    ]
)
classifier_chain = tagging_prompt | llm.with_structured_output(MemoryTaggingSchema)

# @app.on_event("startup")
# async def startup_event():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)


async def get_or_create_session(
    db: AsyncSession, session_id: str, first_message: str
) -> ChatSession:
    session_result = await db.execute(
        select(ChatSession).where(ChatSession.id == session_id)
    )
    chat_session = session_result.scalar_one_or_none()
    if not chat_session:
        chat_session = ChatSession(id=session_id, title=first_message[:30], summary="")
        db.add(chat_session)
        await db.commit()
    return chat_session


async def populate_state_context(db: AsyncSession, state: AgentState) -> list[Message]:
    msg_result = await db.execute(
        select(Message)
        .where(Message.session_id == uuid.UUID(state.session_id))
        .order_by(Message.created_at.asc())
    )
    db_messages = list(msg_result.scalars().all())

    important_msgs = [msg for msg in db_messages if msg.is_Important]
    WINDOW_SIZE = 6

    recent_db_messages = (
        db_messages[-WINDOW_SIZE:] if len(db_messages) > WINDOW_SIZE else db_messages
    )

    combined_messages = list(dict.fromkeys(important_msgs + list(recent_db_messages)))
    combined_messages.sort(key=lambda x: x.created_at)

    for msg in combined_messages:
        if msg.role == "user":
            state.messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            state.messages.append(
                AIMessage(content=msg.content or "", tool_calls=msg.tool_calls or [])
            )
        elif msg.role == "tool":
            t_id = "legacy_tool_call_id"
            if isinstance(msg.tool_calls, dict):
                t_id = msg.tool_calls.get("id", t_id) or msg.tool_calls.get(
                    "tool_call_id", t_id
                )
            state.messages.append(ToolMessage(content=msg.content, tool_call_id=t_id))

    return db_messages


async def compress_old_history_background(chat_session: ChatSession, db_messages: list):
    WINDOW_SIZE = 6
    if len(db_messages) <= WINDOW_SIZE:
        return
    older_messages = db_messages[:-WINDOW_SIZE]
    unimportant_older_messages = [m for m in older_messages if not m.is_Important]
    if not unimportant_older_messages:
        return

    formatted_old_chat = "\n".join(
        [f"{m.role}: {m.content}" for m in unimportant_older_messages]
    )
    summary_prompt = (
        f"Progressively summarize the following chat transcript lines concisely while keeping "
        f"key developer configurations or constraints intact. Core historical context:\n"
        f"Existing Summary: {chat_session.summary or 'None'}\n\n"
        f"New Transcript lines to merge:\n{formatted_old_chat}\n\n"
        f"New Summary:"
    )
    try:
        summary_response = await llm.ainvoke(summary_prompt)
        chat_session.summary = str(summary_response.content)
    except Exception as e:
        print(f"Non-blocking summary failure: {e}")

#the function which actauly routes the tools based on the user message and the current state of the agent and this is called in the main loop of the agent and based on the user message and the current state of the agent it will select the tools from the registry and bind them to the llm for that iteration
async def semantic_tool_router(user_text: str, current_state: str) -> List[Any]:
    """
    Scans the global TOOL_REGISTRY and dynamically selects the best tools for the current prompt.
    """
    # 1. Build a list of all available tools and their descriptions for the LLM to read
    registry_info = "\n".join([f"- Name: {name} | Description: {tool.description}" for name, tool in TOOL_REGISTRY.items()])

    # 2. The Strict Routing Prompt
    # 2. The Strict Routing Prompt
    routing_prompt = f"""
    CRITICAL SYSTEM INSTRUCTION: You are a strict JSON routing controller.
    The user is currently in state: {current_state}.
    User message: "{user_text}"
    
    Global Tool Registry:
    {registry_info}
    
    INSTRUCTIONS:
    1. ALWAYS include 'change_behaviour_profile' in your list so the agent can switch states.
    2. Select up to 2 additional tools that are highly relevant to the User message.
    
    FINAL WARNING: You MUST output pure JSON. Do not add any extra keys.
    Your output MUST exactly match this JSON structure:
    {{"selected_tools": ["change_behaviour_profile", "other_tool"]}}
    """

    try:
        # 3. Use the fast 8B model in JSON mode to act as the router
        router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=SecretStr(api_key))
        router = router_llm.with_structured_output(ToolSelection, method="json_mode")
        
        selection = await router.ainvoke(routing_prompt)
        selected_tool_names = selection.selected_tools
        print(f"🧠 [ROUTER] Selected tools: {selected_tool_names}")
        
        # 4. Map the string names back to actual LangChain Tool objects
        active_tools = []
        for name in selected_tool_names:
            if name in TOOL_REGISTRY:
                active_tools.append(TOOL_REGISTRY[name])
                
        # Failsafe: Ensure transition tool is always present
        if TOOL_REGISTRY["change_behaviour_profile"] not in active_tools:
            active_tools.append(TOOL_REGISTRY["change_behaviour_profile"])
            
        return active_tools

    except Exception as e:
        print(f"⚠️ Router failed, falling back to safe defaults: {e}")
        return [TOOL_REGISTRY["change_behaviour_profile"]]
    
@app.post("/api/agent")
async def user_message(payload: UserMessage, db: AsyncSession = Depends(get_db)):
    session_uuid = uuid.UUID(payload.sessionId)
    chat_session = await get_or_create_session(db, str(session_uuid), payload.message)

    state = AgentState(
        session_id=payload.sessionId, current_summary=chat_session.summary or ""
    )

    db_messages = await populate_state_context(db, state)

    try:
        user_classification = await classifier_chain.ainvoke(
            {"content": payload.message}
        )
        state.is_user_msg_important = user_classification.is_Important
    except Exception:
        state.is_user_msg_important = False
    state.messages.append(HumanMessage(content=payload.message))
    db.add(
        Message(
            session_id=session_uuid,
            role="user",
            content=payload.message,
            is_Important=state.is_user_msg_important,
        )
    )

    # =====================================================================
    # 🧠 BACKGROUND TASK: EXTRACT KNOWLEDGE GRAPH (Runs silently)
    # =====================================================================
    async def extract_and_save_graph(user_text: str):
        try:
            # Upgrade to the much smarter 70B model to stop XML/Schema hallucinations
            extractor_llm_graph = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=SecretStr(api_key))
            
            # Remove method="json_mode" so LangChain securely passes the exact Pydantic schema
            graph_extractor = extractor_llm_graph.with_structured_output(KnowledgeGraphUpdate)
            
            # Keep the prompt clean and simple
            graph_update = await graph_extractor.ainvoke(
                f"Extract core facts as nodes and relationships from this message: '{user_text}'"
            )
            await neo4j_client.execute_graph_update(graph_update.model_dump())
            print("✅ Knowledge Graph successfully updated in the background.")
        except Exception as e:
            print(f"⚠️ Graph extraction failed, skipping memory update: {str(e)}")

    # Fire and forget! Extracts the graph while the main loop starts
    asyncio.create_task(extract_and_save_graph(payload.message))

    # =====================================================================
    # 🔍 FETCH PERMANENT TRUTH (Await before the LLM thinks)
    # =====================================================================
    permanent_facts_text = await neo4j_client.fetch_user_graph_facts()


    # =====================================================================
    # 🔄 STEP 3: THE REACT SYSTEM RUNS FULLY IN STRINGS
    # =====================================================================
    BEHAVIOUR_MAP = {
        BehaviourPattern.CASUAL_PRODUCTIVITY: {
            "prompt": (
                """You are Nexus, in CASUAL_PRODUCTIVITY mode. 

[EPHEMERAL CONTEXT - RECENT CONVERSATION SUMMARY]
{summary}

[PERMANENT FACTS - GRAPH MEMORY]
WARNING: The facts below are the absolute truth about the user. They OVERRIDE any conflicting information found in the conversational summary above.
{permanent_memory}

INSTRUCTIONS:
Chat casually. Do not use complex tools. If the user asks for deep research or coding, use the change_behaviour_profile tool immediately."""
            )
        },
        BehaviourPattern.DEEP_RESEARCH: {
            "prompt": (
                """You are an elite Research Analyst. Use tools to gather deep evidentiary logs.

[EPHEMERAL CONTEXT - RECENT CONVERSATION SUMMARY]
{summary}

[PERMANENT FACTS - GRAPH MEMORY]
WARNING: The facts below are the absolute truth about the user. They OVERRIDE any conflicting information found in the conversational summary above.
{permanent_memory}"""
            )
        },
        BehaviourPattern.STANDARD_CODING: {
            "prompt": (
                """You are a senior Software Engineer. Focus purely on clean, production-grade code syntax.

[EPHEMERAL CONTEXT - RECENT CONVERSATION SUMMARY]
{summary}

[PERMANENT FACTS - GRAPH MEMORY]
WARNING: The facts below are the absolute truth about the user. They OVERRIDE any conflicting information found in the conversational summary above.
{permanent_memory}"""
            )
        },
    }

    # 1. INITIAL PROMPT INJECTION
    initial_config = BEHAVIOUR_MAP.get(
        state.current_behaviour, BEHAVIOUR_MAP[BehaviourPattern.CASUAL_PRODUCTIVITY]
    )
    
    # 🚨 CRITICAL: Format the string to inject the real variables!
    formatted_initial_prompt = initial_config["prompt"].format(
        summary=state.current_summary,
        permanent_memory=permanent_facts_text
    )

    if state.messages and isinstance(state.messages[0], SystemMessage):
        state.messages[0] = SystemMessage(content=formatted_initial_prompt)
    else:
        state.messages.insert(0, SystemMessage(content=formatted_initial_prompt))

    # 2. LOCK ARRAY LENGTH *AFTER* PROMPT INJECTION!
    input_length = len(state.messages)
    iteration = 0
    ai_response = ""
    
    while iteration < MAX_ITERATIONS:
        # 3. DYNAMIC RE-EVALUATION
        current_config = BEHAVIOUR_MAP.get(
            state.current_behaviour, BEHAVIOUR_MAP[BehaviourPattern.CASUAL_PRODUCTIVITY]
        )

        # 🚨 CRITICAL: Format the string inside the loop as well!
        formatted_loop_prompt = current_config["prompt"].format(
            summary=state.current_summary,
            permanent_memory=permanent_facts_text
        )

        # Force-update index 0 in case the state changed on the last iteration
        state.messages[0] = SystemMessage(content=formatted_loop_prompt)


        # 🚀 PHASE 4: DYNAMIC TOOL ROUTING
        # Instead of pulling from current_config["tools"], we dynamically fetch them!
        active_tools = await semantic_tool_router(
            user_text=payload.message, 
            current_state=state.current_behaviour.value
        )
        # 4. RE-BIND TOOLS DYNAMICALLY
        current_llm = llm.bind_tools(active_tools) if active_tools else llm

        ai_message = await current_llm.ainvoke(state.messages)
        state.messages.append(
            ai_message
        )  # the message here ai sends is that it has some informaiot it should cal some tools or it should just answwer the question so in the ReAct way

        if not ai_message.tool_calls:
            ai_response = ai_message.content
            break

        primary_tool_call = ai_message.tool_calls[0]
        tool_name = primary_tool_call["name"]
        tool_args = primary_tool_call["args"]
        tool_call_id = primary_tool_call["id"]

        # 🚀 THE STATE MUTATOR INTERCEPTOR
        if tool_name == "change_behaviour_profile":
            requested_state = tool_args.get("new_profile")
            print(
                f"🚀 [STATE MUTATION] Agent requested transition to: {requested_state}"
            )

            if requested_state in BehaviourPattern.__members__:
                state.current_behaviour = BehaviourPattern[requested_state]

            tool_message = ToolMessage(
                content=f"System context mutated to {state.current_behaviour.value}. You now have access to new tools.",
                tool_call_id=tool_call_id,
                status="success",
            )
            state.messages.append(tool_message)
            iteration += 1

            # RESTART LOOP - Steps 3 & 4 will now grab the new DEEP_RESEARCH tools!
            continue

        # 🎯 DYNAMIC TOOL EXECUTION FIREWALL
        # 🎯 DYNAMIC TOOL EXECUTION FIREWALL
        allowed_tool_names = [t.name for t in active_tools]
        
        if tool_name not in allowed_tool_names:
            tool_message = ToolMessage(
                content=f"ERROR: Access Denied. Tool '{tool_name}' is forbidden under your current state ({state.current_behaviour.value}).",
                tool_call_id=tool_call_id,
                status="error",
            )
            state.messages.append(tool_message)
            iteration += 1
            
        elif tool_name in TOOL_REGISTRY:
            # 🛑 PHASE 5: HUMAN-IN-THE-LOOP INTERCEPTOR
            if tool_name in REQUIRES_APPROVAL_TOOLS:
                print(f"🛑 [HIBERNATION TRIGGERED] Tool '{tool_name}' requires human approval.")
                
                # 1. Save the exact tool execution request to our database so we can resume it later
                state.new_agent_messages = state.messages[input_length:]
                
                # (You would normally run your db.add() loop here to save the exact state before sleeping)
                
                # 2. Break the loop and return the special Approval Payload directly to the frontend
                return {
                    "status": "requires_approval",
                    "session_id": payload.sessionId,
                    "pending_action": {
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                        "tool_call_id": tool_call_id
                    },
                    "message": f"The agent wants to execute {tool_name}. Do you approve?"
                }

            # If it's a safe tool (like web search), execute it normally
            tool_message = await safe_execute_tool(
                TOOL_REGISTRY[tool_name], tool_args, tool_call_id
            )
            state.messages.append(tool_message)
            iteration += 1
            
        else:
            tool_message = ToolMessage(
                content=f"ERROR: Tool missing.",
                tool_call_id=tool_call_id,
                status="error",
            )
            state.messages.append(tool_message)
            iteration += 1

    # FINAL CHECK: If we exit the loop due to max iterations but never got a clean response, we take the last AI message content as the fallback answer to ensure we always return something meaningful to the user.
    if not ai_response:
        for msg in reversed(state.messages[input_length:]):
            if isinstance(msg, AIMessage) and msg.content:
                ai_response = msg.content
                break
    # =====================================================================
    # 🎯 STEP 4: THE GATEWAY EXTRACTION LAYER (The Concept in Action)
    # =====================================================================
    # We serialize what the agent just performed over the last few rounds into a clear string report.
    # This report contains the raw tools data strings + final text thoughts.
    recent_execution_history = "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in state.messages[input_length:]]
    )
    extraction_instruction = (
        f"CRITICAL SYSTEM INSTRUCTION: You are a strict JSON data parser, NOT a chatbot. "
        f"DO NOT include conversational filler, greetings, or explanations. "
        f"DO NOT output phrases like 'Here are the links' or 'Here is your table'.\n\n"
        
        f"Analyze the following operational execution logs. Break down the textual narrative thoughts, "
        f"tool observation reports, and metrics into an ordered sequence of visual UI component block elements matching these strict layout rules:\n\n"
        
        f"1. For 'markdown_text' blocks, supply your written response ONLY in the 'markdown_text' field.\n"
        f"2. For 'data_table' blocks, you must design a complete grid. Define the columns with unique ID values, "
        f"and map the rows as objects where the keys EXACTLY MATCH your column IDs. "
        f"Example: If column IDs are ['Category', 'Pros', 'Cons'], then your rows MUST look like: "
        f"[{{'Category': 'Definition', 'Pros': 'Scalability', 'Cons': 'Complexity'}}]. \n\n"
        
        f"Logs to process:\n{recent_execution_history}\n\n"
        f"FINAL WARNING: Output ONLY pure, valid JSON matching the schema. Any other text will crash the system."
    )
    try:
        # The extraction model reads the string logs and structures them perfectly into our UI pipeline model.
        # It doesn't matter if we have 2 tools or 200 tools, the model categorizes strings into generic UI elements.
        final_structured_payload = await structured_extractor.ainvoke(
            extraction_instruction
        )
    except Exception as e:
        # Resiliency Guardrail: Fallback immediately to a single plain markdown text block if extraction fails
        safe_text = str(ai_response) if ai_response else "Processing complete."
        print(f"Extraction Layer Fallback Triggered: {str(e)}")
        final_structured_payload = ScalableAgentResponseSchema(
            ui_pipeline=[UIBlock(block_type="markdown_text", markdown_text=safe_text)]
        )

    # =====================================================================
    # 📝 STEP 5: EXTRACT AND PERSIST TRANSACTION HISTORY (Unchanged)
    # =====================================================================
    state.new_agent_messages = state.messages[input_length:]

    # We loop through the new messages generated in this session and persist them with proper role tagging and tool call metadata. This ensures our database reflects the full context of the agent's thought process, tool usage, and final response.
    for msg in state.new_agent_messages:
        # 1. Default variables for this specific iteration
        content_payload = str(msg.content) if msg.content else ""
        t_calls = None
        role_type = None

        # 2. Safely route and process based on explicit types
        if isinstance(msg, AIMessage):
            role_type = "assistant"
            t_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None

            # Run AI memory tagging only for assistant messages with content
            if msg.content:
                try:
                    ai_classification = await classifier_chain.ainvoke(
                        {"content": msg.content}
                    )
                    state.is_ai_msg_important = ai_classification.is_Important
                except Exception:
                    state.is_ai_msg_important = False

        elif isinstance(msg, ToolMessage):
            role_type = "tool"
            t_calls = {"id": getattr(msg, "tool_call_id", "legacy_tool_call_id")}
            state.is_ai_msg_important = False

        else:
            continue

        # 4. Safely commit the strictly validated variables
        db.add(
            Message(
                session_id=session_uuid,
                role=role_type,
                content=content_payload,
                tool_calls=t_calls,
                is_Important=state.is_ai_msg_important,
            )
        )
    await db.flush()
    await compress_old_history_background(chat_session, db_messages)
    await db.commit()

    # =====================================================================
    # 🚀 STEP 6: SEND THE NEW STRUCTURED BLOCK PAYLOAD TO FRONTEND
    # =====================================================================
    # Instead of just sending raw string '{"response": ai_response}', we return
    # the entire type-safe JSON object configuration map!
    return {
        "status": "success",
        "session_id": payload.sessionId,
        "current_summary": state.current_summary,
        "data": final_structured_payload.model_dump(),
    }


@app.post("/api/agent/resume")
async def resume_agent(payload: ResumeAction, db: AsyncSession = Depends(get_db)):
    """
    Catches the human's approval/rejection, executes the pending tool (if approved),
    and resumes the ReAct loop.
    """
    session_uuid = uuid.UUID(payload.sessionId)
    chat_session = await get_or_create_session(db, str(session_uuid), "Resumed Session")

    # 1. Rebuild the Agent's Brain from PostgreSQL
    state = AgentState(
        session_id=payload.sessionId, current_summary=chat_session.summary or ""
    )
    db_messages = await populate_state_context(db, state)
    input_length = len(state.messages)

    # 2. INJECT THE HUMAN'S DECISION
    if payload.is_approved:
        print(f"✅ [HUMAN APPROVED] Executing paused tool: {payload.tool_name}")
        # Actually execute the tool now!
        tool_message = await safe_execute_tool(
            TOOL_REGISTRY[payload.tool_name], payload.tool_args, payload.tool_call_id
        )
    else:
        print(f"❌ [HUMAN REJECTED] Cancelling tool: {payload.tool_name}")
        # Inject a strict failure so the LLM knows it was denied
        tool_message = ToolMessage(
            content=f"ERROR: The human explicitly REJECTED this action. Do not try again. Apologize and ask what they would like to do instead.",
            tool_call_id=payload.tool_call_id,
            status="error"
        )
        
    state.messages.append(tool_message)

    # 3. RESUME THE REACT LOOP
    permanent_facts_text = await neo4j_client.fetch_user_graph_facts()
    iteration = 0
    ai_response = ""

    while iteration < MAX_ITERATIONS:
        # Dynamically fetch tools so the agent still has capabilities
        active_tools = await semantic_tool_router(
            user_text="User just responded to an approval request.", 
            current_state=state.current_behaviour.value
        )
        current_llm = llm.bind_tools(active_tools) if active_tools else llm

        # The LLM wakes up, reads the human's decision, and decides what to say next!
        ai_message = await current_llm.ainvoke(state.messages)
        state.messages.append(ai_message)

        if not ai_message.tool_calls:
            ai_response = ai_message.content
            break

        # If it tries to use MORE tools, handle them safely (standard firewall)
        primary_tool_call = ai_message.tool_calls[0]
        tool_name = primary_tool_call["name"]
        tool_args = primary_tool_call["args"]
        tool_call_id = primary_tool_call["id"]
        
        if tool_name in TOOL_REGISTRY:
            tool_msg = await safe_execute_tool(TOOL_REGISTRY[tool_name], tool_args, tool_call_id)
            state.messages.append(tool_msg)
        iteration += 1

    if not ai_response:
        for msg in reversed(state.messages[input_length:]):
            if isinstance(msg, AIMessage) and msg.content:
                ai_response = msg.content
                break

    # 4. RUN THE UI EXTRACTOR (Exactly the same as your main route)
    recent_execution_history = "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in state.messages[input_length:]]
    )
    extraction_instruction = (
        f"CRITICAL SYSTEM INSTRUCTION: You are a strict JSON data parser. Output pure JSON matching the schema.\n\n"
        f"Logs to process:\n{recent_execution_history}"
    )
    try:
        final_structured_payload = await structured_extractor.ainvoke(extraction_instruction)
    except Exception as e:
        final_structured_payload = ScalableAgentResponseSchema(
            ui_pipeline=[UIBlock(block_type="markdown_text", markdown_text=str(ai_response))]
        )

    # 5. SAVE NEW MESSAGES TO DB
    state.new_agent_messages = state.messages[input_length:]
    for msg in state.new_agent_messages:
        content_payload = str(msg.content) if msg.content else ""
        t_calls = None
        role_type = None

        if isinstance(msg, AIMessage):
            role_type = "assistant"
            t_calls = msg.tool_calls if hasattr(msg, "tool_calls") else None
        elif isinstance(msg, ToolMessage):
            role_type = "tool"
            t_calls = {"id": getattr(msg, "tool_call_id", "legacy_tool_call_id")}
        else:
            continue

        db.add(Message(
            session_id=session_uuid, role=role_type, content=content_payload, tool_calls=t_calls, is_Important=False
        ))
        
    await db.commit()

    # 6. RETURN FINAL PAYLOAD TO FRONTEND
    return {
        "status": "success",
        "session_id": payload.sessionId,
        "current_summary": state.current_summary,
        "data": final_structured_payload.model_dump(),
    }