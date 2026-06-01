from enum import Enum
import os
import uuid
from datetime import datetime
from typing import List, Any, Optional
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

# LangChain Core Messaging Ecosystem
from langchain_groq import ChatGroq  
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate

# Native Database Dependencies
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .database import get_db, engine, Base
from .models import ChatSession, Message

# Specialized Agent Tools & Base Configuration
from .prompts import SYSTEM_PROMPT

# FIXED: Correct clean registry and safe execution wrapper import path
from .tools.registry import TOOL_REGISTRY, safe_execute_tool

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    current_behaviour: BehaviourPattern = Field(default=BehaviourPattern.CASUAL_PRODUCTIVITY)

class UserMessage(BaseModel):
    message: str
    sessionId: str 

api_key = os.environ["GROQ_API_KEY"]
tools = list(TOOL_REGISTRY.values())
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=SecretStr(api_key)
)

MAX_ITERATIONS = 5
llm_with_tools = llm.bind_tools(tools)

class MemoryTaggingSchema(BaseModel):
    is_Important: bool = Field(description="True context if text metadata should be permanently pinned.")

tagging_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an elite database state memory optimization controller. Evaluate the structural value of the text."),
    ("human", "Analyze the following content and extract if it should be marked as permanently important:\n\nContent: {content}")
])
classifier_chain = tagging_prompt | llm.with_structured_output(MemoryTaggingSchema)

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_or_create_session(db: AsyncSession, session_id: str, first_message: str) -> ChatSession:
    session_result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    chat_session = session_result.scalar_one_or_none()
    if not chat_session:
        chat_session = ChatSession(id=session_id, title=first_message[:30], summary="")
        db.add(chat_session)
        await db.commit()
    return chat_session

async def populate_state_context(db: AsyncSession, state: AgentState) -> list[Message]:
    msg_result = await db.execute(
        select(Message).where(Message.session_id == uuid.UUID(state.session_id)).order_by(Message.created_at.asc())
    )
    db_messages = list(msg_result.scalars().all())
    
    important_msgs = [msg for msg in db_messages if msg.is_Important]
    WINDOW_SIZE = 6

    state.messages.append(SystemMessage(content=SYSTEM_PROMPT.format(summary=state.current_summary)))
    recent_db_messages = db_messages[-WINDOW_SIZE:] if len(db_messages) > WINDOW_SIZE else db_messages
    
    combined_messages = list(dict.fromkeys(important_msgs + list(recent_db_messages)))
    combined_messages.sort(key=lambda x: x.created_at)

    for msg in combined_messages:
        if msg.role == "user":
            state.messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            state.messages.append(AIMessage(content=msg.content or "", tool_calls=msg.tool_calls or []))
        elif msg.role == "tool":
            t_id = "legacy_tool_call_id"
            if isinstance(msg.tool_calls, dict):
                t_id = msg.tool_calls.get("id", t_id) or msg.tool_calls.get("tool_call_id", t_id)
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
        
    formatted_old_chat = "\n".join([f"{m.role}: {m.content}" for m in unimportant_older_messages])
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

@app.post("/api/agent")
async def user_message(payload: UserMessage, db: AsyncSession = Depends(get_db)):
    session_uuid = uuid.UUID(payload.sessionId)
    chat_session = await get_or_create_session(db, str(session_uuid), payload.message)

    state = AgentState(
        session_id=payload.sessionId,
        current_summary=chat_session.summary or ""
    )

    db_messages = await populate_state_context(db, state)

    try:
        user_classification = await classifier_chain.ainvoke({"content": payload.message})
        state.is_user_msg_important = user_classification.is_Important
    except Exception:
        state.is_user_msg_important = False

    state.messages.append(HumanMessage(content=payload.message))
    db.add(Message(session_id=session_uuid, role="user", content=payload.message, is_Important=state.is_user_msg_important))

    input_length = len(state.messages)
    iteration = 0
    ai_response = ""
    # =====================================================================
    # 🔄 STEP 5: TRUE SEQUENTIAL REACT WORKFLOW LOOP
    # =====================================================================
    iteration = 0
    ai_response = ""

    while iteration < MAX_ITERATIONS:
        # 1. LLM evaluates the entire timeline and generates its next decision
        ai_message = await llm_with_tools.ainvoke(state.messages)
        state.messages.append(ai_message)

        # 2. If the LLM didn't call a tool, it has reached its final answer!
        if not ai_message.tool_calls:
            ai_response = ai_message.content
            break

        # 3. CRITICAL REACT ARCHITECTURE: Enforce processing exactly ONE tool call at a time.
        # Even if the LLM hallucinates or outputs multiple tool calls, we only execute the first one.
        # This forces the agent to look at the observation before making its next move.
        primary_tool_call = ai_message.tool_calls[0]
        tool_name = primary_tool_call["name"]
        tool_args = primary_tool_call["args"]
        tool_call_id = primary_tool_call["id"]

        if tool_name in TOOL_REGISTRY:
            tool_message = await safe_execute_tool(
                tool_function=TOOL_REGISTRY[tool_name],
                tool_input=tool_args,
                tool_call_id=tool_call_id
            )
        else:
            # Handle bad tools cleanly inside our immutable timeline
            tool_message = ToolMessage(
                content=f"ERROR: Tool '{tool_name}' does not exist inside registry. Choose from {list(TOOL_REGISTRY.keys())}.",
                tool_call_id=tool_call_id,
                status="error"
            )

        # 4. Append this single observation directly to the conversation state
        state.messages.append(tool_message)
        
        # 5. Spin the loop back. The LLM will now read this ToolMessage result,
        # reason about it, and then decide whether it needs another tool or can answer.
        iteration += 1

    if not ai_response:
        for msg in reversed(state.messages[input_length:]):
            if isinstance(msg, AIMessage) and msg.content:
                ai_response = msg.content
                break

    # =====================================================================
    # 📝 STEP 6: EXTRACT AND PERSIST TRANSACTION HISTORY
    # =====================================================================
    state.new_agent_messages = state.messages[input_length:]

    for msg in state.new_agent_messages:
        role_type = "assistant" if isinstance(msg, AIMessage) else "tool"
        content_payload = str(msg.content) if msg.content else ""
        t_calls = None
        
        if role_type == "assistant":
            t_calls = msg.tool_calls if hasattr(msg, 'tool_calls') else None
            if msg.content:
                try:
                    ai_classification = await classifier_chain.ainvoke({"content": msg.content})
                    state.is_ai_msg_important = ai_classification.is_Important
                except Exception:
                    state.is_ai_msg_important = False
        elif role_type == "tool":
            # Mirrors parse structure inside populate_state_context perfectly
            t_calls = {"id": getattr(msg, "tool_call_id", "legacy_tool_call_id")}

        db.add(Message(
            session_id=session_uuid, 
            role=role_type, 
            content=content_payload, 
            tool_calls=t_calls,
            is_Important=state.is_ai_msg_important if role_type == "assistant" else False
        ))

    await db.flush() 
    await compress_old_history_background(chat_session, db_messages)
    await db.commit() 

    return {"response": ai_response}