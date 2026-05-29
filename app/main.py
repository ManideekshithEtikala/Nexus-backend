import os
from fastapi import FastAPI, Depends
from langchain_groq import ChatGroq  
from langchain_core.tools import tool  
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from .prompts import SYSTEM_PROMPT
from pydantic import SecretStr
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from fastapi.middleware.cors import CORSMiddleware

# Native Database Dependencies
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from .database import get_db, engine, Base
from .models import ChatSession, Message

# Specialized Agent Tools
from .tools import research_docs_tool
from langchain.agents import create_agent

load_dotenv()
app = FastAPI()

class UserMessage(BaseModel):
    message: str
    sessionId: str # Enforce JavaScript camelCase to match the frontend JSON body key perfectly

api_key = os.environ["GROQ_API_KEY"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@tool
async def hi_tool() -> str:
    """Call this tool when the user greets you with words like hi, hello, or hey."""
    return "Hello! How can I assist with you today?"

tools = [hi_tool, research_docs_tool]
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
    api_key=SecretStr(api_key)
)

agent_executor = create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    debug=True
)
# =====================================================================
# 🎛️ NEW: HIGH-PERFORMANCE STRUCTURED CLASSIFICATION CORES
# =====================================================================

class MemoryTaggingSchema(BaseModel):
    """Schema optimized for parameter extraction to determine memory importance."""
    is_Important: bool = Field(
        description="True ONLY if the text defines permanent project parameters, database schemas, "
                    "constants, configuration preferences, or explicit coding patterns. "
                    "False for small talk, intermediate logs, questions, or casual chat phrases."
    )

# A tight, explicit prompt designed to make the extraction super deterministic
tagging_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an elite database state memory optimization controller. Evaluate the structural value of the text."),
    ("human", "Analyze the following content and extract if it should be marked as permanently important:\n\nContent: {content}")
])

# Bind the Pydantic schema to your LLM engine to force a structured JSON output response
classifier_chain = tagging_prompt | llm.with_structured_output(MemoryTaggingSchema)

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        # Automatically registers, patches, or generates your schemas directly in Aiven Cloud
        await conn.run_sync(Base.metadata.create_all)


# =====================================================================
# 🛠️ DECOUPLED CORE DATABASE & AGENT SERVICES
# =====================================================================

async def get_or_create_session(db: AsyncSession, session_id: str, first_message: str) -> ChatSession:
    """Resolves and retrieves a tracking session, initializing it if missing."""
    session_result = await db.execute(select(ChatSession).where(ChatSession.id == session_id))
    chat_session = session_result.scalar_one_or_none()

    if not chat_session:
        chat_session = ChatSession(id=session_id, title=first_message[:30], summary="")
        db.add(chat_session)
        await db.commit()
    
    return chat_session


async def build_optimized_context(db: AsyncSession, session_id: str, current_summary: str | None) -> tuple[list, list]:
    """Fetches full database message logs, applies the sliding window filter, 

    and translates logs into LangChain memory objects.
    """
    # 1. Fetch all messages for the session, sorted by creation time
    msg_result = await db.execute(
        select(Message).where(Message.session_id == session_id).order_by(Message.created_at.asc())
    )
    db_messages:list[Message] =list(msg_result.scalars().all()) 
    #getting only impoartant mesages from the database which are marked as important by the agent during the execution of the graph. This is done to ensure that the important messages are always included in the context window, even if they are older than the sliding window size.
    important_msgs = [msg for msg in db_messages if msg.is_Important]

    WINDOW_SIZE = 6
    history_stack = []

    # 1. If a summary exists, inject it as a system message at the base of the stack to preserve long-term context without consuming window tokens
    if current_summary:
        summary_instruction = f"Summary of the conversation so far: {current_summary}"
        history_stack.append(SystemMessage(content=summary_instruction))

    # 2. Extract only the recent window slice
    recent_db_messages = db_messages[-WINDOW_SIZE:] if len(db_messages) > WINDOW_SIZE else db_messages
    # Combine important messages with the recent window, ensuring no duplicates and maintaining chronological order
    combined_messages = list(dict.fromkeys(important_msgs + list(recent_db_messages)))
    combined_messages.sort(key=lambda x: x.created_at)
    # 3. Translate database message records into LangChain message objects for the agent's working memory
    for msg in combined_messages:
        if msg.role == "user":
            history_stack.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            history_stack.append(AIMessage(content=msg.content, tool_calls=msg.tool_calls or []))
        elif msg.role == "tool":
            history_stack.append(ToolMessage(content=msg.content, tool_call_id=msg.content))

    return history_stack, db_messages


async def compress_old_history_background(db: AsyncSession, chat_session: ChatSession, db_messages: list):
    """Summarizes older conversation turns that slid past the current active window constraint."""
    WINDOW_SIZE = 6
    if len(db_messages) <= WINDOW_SIZE:
        return

    # Isolate messages outside our sliding active window boundary
    older_messages = db_messages[:-WINDOW_SIZE]
    #if the messgae is importatnt only then we will add that to the summary if not then it is not added to the summary 
    important_older_messages = [m for m in older_messages if m.is_Important]
    if not important_older_messages:
        return
    formatted_old_chat = "\n".join([f"{m.role}: {m.content}" for m in important_older_messages])
    
    summary_prompt = (
        f"Progressively summarize the following chat transcript lines concisely while keeping "
        f"key developer configurations or constraints intact. Core historical context:\n"
        f"Existing Summary: {chat_session.summary or 'None'}\n\n"
        f"New Transcript lines to merge:\n{formatted_old_chat}\n\n"
        f"New Summary:"
    )
    
    try:
        summary_response = await llm.ainvoke(summary_prompt)
        # 💥 THE FIX: Force the content to be a clean string
        if isinstance(summary_response.content, list):
            # If it's a list of blocks, join them together or extract the text component
            chat_session.summary = str(summary_response.content[0] if summary_response.content else "")
        else:
            chat_session.summary = str(summary_response.content)
            
    except Exception as e:
        print(f"Non-blocking background context summary compression failure: {e}")


# =====================================================================
# 🚀 CLEAN, AGGREGATED ENDPOINT ROUTER
# =====================================================================

@app.post("/api/agent")
async def user_message(payload: UserMessage, db: AsyncSession = Depends(get_db)):
    session_uuid = payload.sessionId

    # Step 1: Ensure ChatSession context exists
    chat_session = await get_or_create_session(db, session_uuid, payload.message)

    # Step 2: Build the Optimized Memory History Stack (Sliding Window applied here)
    history_stack, db_messages = await build_optimized_context(db, session_uuid, chat_session.summary)
    try:
        user_classification = await classifier_chain.ainvoke({"content": payload.message})
        is_user_msg_important = user_classification.is_Important
    except Exception:
        is_user_msg_important = False
    # Step 3: Append the incoming turn to our working context array and write to database
    history_stack.append(HumanMessage(content=payload.message))
    db.add(Message(session_id=session_uuid, role="user", content=payload.message,is_Important=is_user_msg_important))

    # Step 4: Fire the agent executor orchestration graph
    result = await agent_executor.ainvoke({"messages": history_stack})
    
    # Step 5: Isolate and persist new execution turns (assistant text responses and tool telemetry)
    new_messages = result["messages"][len(history_stack):]
    ai_response = ""
    
    for msg in new_messages:
        role_type = "assistant" if type(msg).__name__ == "AIMessage" else "tool"
        content_payload = msg.content
        t_calls = msg.tool_calls if hasattr(msg, 'tool_calls') else None
        
        if role_type == "assistant" and content_payload:
            ai_response = content_payload
        is_ai_msg_important = False
        if role_type == "assistant" and content_payload:
            try:
                ai_classification = await classifier_chain.ainvoke({"content": content_payload})
                is_ai_msg_important = ai_classification.is_Important
            except Exception:
                is_ai_msg_important = False
        db.add(Message(
            session_id=session_uuid, role=role_type, content=content_payload, tool_calls=t_calls,is_Important=is_ai_msg_important
        ))

    if not ai_response:
        ai_response = result["messages"][-1].content

    await db.flush() # Stage changes, preparing database row counts for the compression calculation

    # Step 6: Compress older historical records if the global array has overflowed our limits
    await compress_old_history_background(db, chat_session, db_messages)

    await db.commit() # Atomic save of all logs and summary status data blocks to Aiven Cloud
    return {"response": ai_response}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)