import os
import uvicorn
from dotenv import load_dotenv

# prompts
from prompts import NEXUS_SYSTEM_PROMPT

# strctured output
from models import parser
from groq import RateLimitError

# LangChain Imports
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import trim_messages
from tools import (
    # File I/O
    read_file,
    read_file_lines,
    write_file,
    append_to_file,
    list_files,
    delete_file,
    copy_file,
    move_file,
    # Terminal
    run_command,
    run_python_code,
    get_running_processes,
    kill_process,
    # Code intelligence
    search_in_files,
    str_replace_in_file,
    get_file_structure,
    count_lines_of_code,
    # Git
    git_status,
    git_diff,
    git_log,
    git_commit,
    # Web
    fetch_url,
    search_pypi,
    # System
    get_environment_info,
    get_env_variable,
    check_disk_usage,
)

tools = [
    read_file,
    read_file_lines,
    write_file,
    append_to_file,
    list_files,
    delete_file,
    copy_file,
    move_file,
    # Terminal
    run_command,
    run_python_code,
    get_running_processes,
    kill_process,
    # Code intelligence
    search_in_files,
    str_replace_in_file,
    get_file_structure,
    count_lines_of_code,
    # Git
    git_status,
    git_diff,
    git_log,
    git_commit,
    # Web
    fetch_url,
    search_pypi,
    # System
    get_environment_info,
    get_env_variable,
    check_disk_usage,
]
# FastAPI Imports
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import asyncio
from fastapi.middleware.cors import CORSMiddleware

# Local Imports
from database.database import async_engine
from models.schemas import ChatRequest
from memory.hierarchical_history import HierarchicalSQLChatMessageHistory

load_dotenv()


# ─── 2. LLM + PROMPT ─────────────────────────────────────────────────────────

llm = ChatGroq(
    model="qwen/qwen3-32b", api_key=os.getenv("API_KEY"), temperature=0, max_retries=3
)

# Short-term Memory trimmer — keeps the last 10 messages only.
# This prevents the context window from growing unbounded with long conversations.
# max_tokens=10 here means "10 message objects", not raw tokens, because token_counter=len.
trimmer = trim_messages(
    max_tokens=6,
    strategy="last",
    token_counter=len,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", NEXUS_SYSTEM_PROMPT),
        (
            "system",
            "You must always provide your FINAL ANSWER in the following JSON format:\n{format_instructions}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# ─── 3. AGENT ────────────────────────────────────────────────────────────────

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

# Trim history BEFORE it is injected into the prompt so we never exceed the context window
agent_with_trimming = (
    RunnablePassthrough.assign(chat_history=lambda x: trimmer.invoke(x["chat_history"]))
    | agent_executor
)

# ─── 4. MEMORY WRAPPER ───────────────────────────────────────────────────────
# Hierarchical memory: last 6 messages raw + rolling bullet-point summary


def get_session_history(session_id: str):
    """Returns hierarchical chat history with summarization."""
    return HierarchicalSQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine,
        table_name="message_store",
        async_mode=True,
    )


agent_with_chat_history = RunnableWithMessageHistory(
    agent_with_trimming,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# ─── 5. FASTAPI SERVER ───────────────────────────────────────────────────────

app = FastAPI(title="Nexus Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"session_id": request.session_id}}

    try:
        # 1. Attempt to invoke the agent
        result = await agent_with_chat_history.ainvoke(
            {"input": request.message}, config=config
        )
    except Exception as e:
        # Check if it's a rate limit error (Groq/LangChain specific)
        error_msg = str(e).lower()
        if "rate_limit_exceeded" in error_msg or "429" in error_msg:
            return {
                "data": {
                    "answer": "System is currently busy due to high demand (Rate Limit Exceeded). Please wait about 60 seconds and try again.",
                    "files_modified": [],
                    "commands_run": [],
                    "confidence": 0,
                },
                "error": "rate_limit",
                "session_id": request.session_id,
            }

        # General Error Handling
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "session_id": request.session_id,
        }

    # 2. Extract the raw text output
    raw_output = result.get("output", "")

    try:
        # 3. Parse the structured data
        structured_data = parser.parse(raw_output)
        return {
            "data": structured_data,
            "session_id": request.session_id,
            "debug_steps": str(result.get("intermediate_steps", [])),
        }
    except Exception as e:
        # Fallback for parsing errors
        return {
            "data": {
                "answer": raw_output,
                "files_modified": [],
                "commands_run": [],
                "confidence": 0.5,
            },
            "error": f"Format error: {str(e)}",
            "session_id": request.session_id,
        }


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat/stream")
async def chat_endpoint_stream(request: ChatRequest):
    """
    Stream agent response in real-time using Server-Sent Events.
    Events:
      - type: "reasoning"  : LLM thinking content
      - type: "tool_call"  : tool invocation
      - type: "observation": tool output
      - type: "answer"     : final answer (final chunk)
      - type: "done"       : completion signal
      - type: "error"      : error message
    """
    config = {"configurable": {"session_id": request.session_id}}
    
    async def event_generator():
        try:
            async for chunk in agent_with_chat_history.astream(
                {"input": request.message}, 
                config=config
            ):
                # Case 1: LLM decided to call tool(s) — includes reasoning
                if isinstance(chunk, dict) and "actions" in chunk:
                    actions = chunk["actions"]
                    for action in actions:
                        # Stream reasoning (thinking) from message_log
                        for msg in getattr(action, "message_log", []):
                            reasoning = getattr(msg, "additional_kwargs", {}).get("reasoning_content", "")
                            if reasoning:
                                yield f"data: {json.dumps({'type': 'reasoning', 'content': reasoning})}\n\n"
                        # Stream tool call
                        yield f"data: {json.dumps({'type': 'tool_call', 'content': {'tool': action.tool, 'input': action.tool_input}})}\n\n"
                
                # Case 2: Tool execution result (observation)
                elif isinstance(chunk, dict) and "steps" in chunk:
                    steps = chunk["steps"]
                    for step in steps:
                        obs = getattr(step, "observation", "")
                        if obs:
                            yield f"data: {json.dumps({'type': 'observation', 'content': str(obs)})}\n\n"
                
                # Case 3: Final answer
                elif "output" in chunk:
                    raw_output = chunk["output"]
                    try:
                        # Parse structured AgentResponse to extract just the answer text
                        structured = parser.parse(raw_output)
                        answer_text = structured.answer  # Pydantic model attribute
                    except Exception:
                        answer_text = raw_output
                    yield f"data: {json.dumps({'type': 'answer', 'content': answer_text})}\n\n"
                    break
                
                # Fallback: unknown chunk type
                else:
                    # Could be AIMessageChunk directly? Try to extract content
                    if hasattr(chunk, 'content') and hasattr(chunk, 'type'):
                        if chunk.type == 'ai':
                            yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
                    elif isinstance(chunk, str):
                        yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
            
            # Done signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    config = {"configurable": {"session_id": request.session_id}}

    try:
        # 1. Attempt to invoke the agent
        result = await agent_with_chat_history.ainvoke(
            {"input": request.message}, config=config
        )
    except Exception as e:
        # Check if it's a rate limit error (Groq/LangChain specific)
        error_msg = str(e).lower()
        if "rate_limit_exceeded" in error_msg or "429" in error_msg:
            return {
                "data": {
                    "answer": "System is currently busy due to high demand (Rate Limit Exceeded). Please wait about 60 seconds and try again.",
                    "files_modified": [],
                    "commands_run": [],
                    "confidence": 0,
                },
                "error": "rate_limit",
                "session_id": request.session_id,
            }

        # General Error Handling
        return {
            "error": f"An unexpected error occurred: {str(e)}",
            "session_id": request.session_id,
        }

    # 2. Extract the raw text output
    raw_output = result.get("output", "")

    try:
        # 3. Parse the structured data
        structured_data = parser.parse(raw_output)
        return {
            "data": structured_data,
            "session_id": request.session_id,
            "debug_steps": str(result.get("intermediate_steps", [])),
        }
    except Exception as e:
        # Fallback for parsing errors
        return {
            "data": {
                "answer": raw_output,
                "files_modified": [],
                "commands_run": [],
                "confidence": 0.5,
            },
            "error": f"Format error: {str(e)}",
            "session_id": request.session_id,
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
