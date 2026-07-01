import os
import uuid
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage
# 🏗️ ARCHITECTURE SERVICES
from app.database import get_db
from app.database.database import DATABASE_URL
from app.database.database_neo4j import neo4j_client
from app.core.schema import UserMessage, ResumeAction, KnowledgeGraphUpdate
from app.core.auth import get_current_user
from app.graph.supervisor_agent.supervisor_graph import build_supervisor_graph as build_graph

# =====================================================================
# ⚙️ SYSTEM INITIALIZATION
# =====================================================================
load_dotenv()

api_key = os.environ["GROQ_API_KEY"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Sets up the AsyncPostgresSaver checkpointer ONCE for the app's lifetime.
    `setup()` creates the checkpointer's tables if they don't exist yet
    (separate from your ChatSession/Message tables).
    """
    from psycopg_pool import AsyncConnectionPool
    from psycopg.rows import dict_row

    # Clean the connection string for psycopg
    checkpointer_url = DATABASE_URL
    if checkpointer_url.startswith("postgresql+asyncpg://"):
        checkpointer_url = checkpointer_url.replace("postgresql+asyncpg://", "postgresql://")
    
    # Use AsyncConnectionPool to automatically handle connection management, reconnects, and scaling.
    # We pass prepare_threshold=None to disable prepared statements (critical for Supabase poolers).
    async with AsyncConnectionPool(
        conninfo=checkpointer_url,
        min_size=0,
        max_size=4,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row}
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://nexus-frontned.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def database_check(db: AsyncSession = Depends(get_db)):
    """Validates connectivity to both PostgreSQL (Supabase) and Neo4j."""
    postgres_status = "pending"
    neo4j_status = "pending"
    postgres_error = None
    neo4j_error = None

    try:
        await asyncio.wait_for(db.execute(text("SELECT 1")), timeout=5.0)
        postgres_status = "connected"
    except asyncio.TimeoutError:
        postgres_status = "timeout"
        postgres_error = "Database query timed out after 5 seconds"
    except Exception as e:
        postgres_status = "failed"
        postgres_error = str(e)

    try:
        await asyncio.wait_for(neo4j_client.test_connection(), timeout=5.0)
        neo4j_status = "connected"
    except asyncio.TimeoutError:
        neo4j_status = "timeout"
        neo4j_error = "Neo4j connection timed out after 5 seconds"
    except Exception as e:
        neo4j_status = "failed"
        neo4j_error = str(e)

    overall_status = (
        "healthy"
        if postgres_status == "connected" and neo4j_status == "connected"
        else "unhealthy"
    )

    return {
        "status": overall_status,
        "postgres": postgres_status,
        "postgres_error": postgres_error,
        "neo4j": neo4j_status,
        "neo4j_error": neo4j_error,
    }


# =====================================================================
# 🛣️ ROUTE 1: PRIMARY AGENT ENDPOINT
# =====================================================================
@app.post("/api/agent")
async def user_message(
    payload: UserMessage,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """
    The main gateway. Builds the agent graph for this request and runs it.
    `thread_id` = session_id, so the checkpointer keeps each session's
    conversation state isolated and resumable.
    """

    # --- Fire-and-forget Neo4j graph extraction (scoped to this user) ---
    async def extract_and_save_graph(user_text: str, owner_user_id: str):
        try:
            extractor_llm_graph = ChatGroq(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.0,
                api_key=SecretStr(api_key),
            )
            graph_extractor = extractor_llm_graph.with_structured_output(
                KnowledgeGraphUpdate,method="json_mode"
            )
            graph_update = await graph_extractor.ainvoke(
                f'''Extract core facts as nodes/edges: '{user_text}' and return in JSON format. and the output shoudl match the format of {KnowledgeGraphUpdate.model_json_schema()}
                CRITICAL instruction for Graph Extraction:
                Every item in the "nodes" array must use the keys "entity_name" and "entity_type". Do NOT use "id" or "text".
                Every item in the "edges" array must use the keys "source_node", "target_node", and "relation". Do NOT use "from" or "to".'''
            )
            await neo4j_client.execute_graph_update(
                graph_update.model_dump(), owner_user_id
            )
        except Exception as e:
            print(f"⚠️ Graph extraction failed: {str(e)}")

    asyncio.create_task(extract_and_save_graph(payload.message, user_id))

    checkpointer = app.state.checkpointer
    compiled_graph = build_graph( checkpointer=checkpointer)

    config = {"configurable": {"thread_id": payload.sessionId, "db": db}}

    initial_state = {
        "messages":[HumanMessage(content=payload.message)],
        "session_id": payload.sessionId,
        "user_id": user_id,
        "user_message": payload.message,
        "routing_round": 0,
        "is_final": False,
        "worker_results": {"__reset__": True},  # Clear any previous results for this session
        "delegated_tasks": {},
    }
    result = await compiled_graph.ainvoke(initial_state, config=config)
    print(f"🟢 [MAIN] full graph result keys -> {list(result.keys())!r}")
    print(
        f"🟢 [MAIN] final_response present? {'final_response' in result!r} "
        f"-> value={result.get('final_response')!r}"
    )

    # Check if the graph paused on an interrupt (e.g. send_email approval)
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        return {
            "status": "requires_approval",
            "session_id": payload.sessionId,
            "pending_action": {
                "tool_name": interrupt_payload["tool_name"],
                "tool_args": interrupt_payload["tool_args"],
                "tool_call_id": interrupt_payload["tool_call_id"],
            },
            "message": interrupt_payload["message"],
        }

    return {
        "status": "success",
        "session_id": payload.sessionId,
        "data": result.get("final_response", "Processing complete."),
    }


# =====================================================================
# 🛣️ ROUTE 2: RESUME HIBERNATED AGENT
# =====================================================================
@app.post("/api/agent/resume")
async def resume_agent(
    payload: ResumeAction,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user),
):
    """
    Resumes a graph that was paused by interrupt() inside the `tools` node.
    """
    checkpointer = app.state.checkpointer
    compiled_graph = build_graph(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": payload.sessionId, "db": db}}

    resume_value = {
        "is_approved": payload.is_approved,
        "tool_name": payload.tool_name,
        "tool_args": payload.tool_args,
        "tool_call_id": payload.tool_call_id,
        
    }

    result = await compiled_graph.ainvoke(Command(resume=resume_value), config=config)

    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        return {
            "status": "requires_approval",
            "session_id": payload.sessionId,
            "pending_action": {
                "tool_name": interrupt_payload["tool_name"],
                "tool_args": interrupt_payload["tool_args"],
                "tool_call_id": interrupt_payload["tool_call_id"],
            },
            "message": interrupt_payload["message"],
        }

    return {
        "status": "success",
        "session_id": payload.sessionId,
        "data": result.get("final_response", "Processing complete."),
    }
