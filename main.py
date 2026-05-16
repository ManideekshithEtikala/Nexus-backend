import os
import json
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# LangChain Imports
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages, AIMessage, HumanMessage, ToolMessage

# Local Imports
from prompts.prompt import NEXUS_SYSTEM_PROMPT, REACT_INJECTION
from models import parser
from database.database import async_engine
from models.schemas import ChatRequest
from memory.hierarchical_history import HierarchicalSQLChatMessageHistory

# Dynamically import all tools from your tools directory package
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

load_dotenv()

# Pack tools into an array and register them to an internal map lookup
# tools_list = [
#     read_file,
#     read_file_lines,
#     write_file,
#     append_to_file,
#     list_files,
#     delete_file,
#     copy_file,
#     move_file,
#     run_command,
#     run_python_code,
#     get_running_processes,
#     kill_process,
#     search_in_files,
#     str_replace_in_file,
#     get_file_structure,
#     count_lines_of_code,
#     git_status,
#     git_diff,
#     git_log,
#     git_commit,
#     fetch_url,
#     search_pypi,
#     get_environment_info,
#     get_env_variable,
#     check_disk_usage,
# ]
# TOOLS_MAP = {t.name: t for t in tools_list}
tools_list = [
    read_file,
    write_file,
    str_replace_in_file,
    get_file_structure,
    run_command,  # This can run Python scripts, git, and system checks anyway!
]
TOOLS_MAP = {t.name: t for t in tools_list}

# ─── 1. LLM CONFIGURATION ───────────────────────────────────────────────────
llm = ChatGroq(
    model="qwen/qwen3-32b", api_key=os.getenv("API_KEY"), temperature=0, max_retries=3
).bind_tools(tools_list)  # Native tool-binding interface

# Memory context optimizer window
trimmer = trim_messages(
    # max_tokens=6,
    # strategy="last",
    # token_counter=len,
    # include_system=True,
    # allow_partial=False,
    # start_on="human",
    max_tokens=4,
    strategy="last",
    token_counter=len,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Core ReAct Prompt Configuration Assembly
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", NEXUS_SYSTEM_PROMPT),
        ("system", REACT_INJECTION),
        (
            "system",
            "You must always provide your FINAL ANSWER in the following JSON format:\n{format_instructions}",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
).partial(format_instructions=parser.get_format_instructions())


# ─── 2. SESSION MEMORY STORAGE LAYER ────────────────────────────────────────
def get_session_history(session_id: str):
    return HierarchicalSQLChatMessageHistory(
        session_id=session_id,
        connection=async_engine,
        table_name="message_store",
        async_mode=True,
    )


# ─── 3. MANUAL CONTROLLED REACT LOOP RUNTIME ────────────────────────────────
async def run_react_loop(
    input_message: str, session_id: str, stream_queue: asyncio.Queue = None
):
    """
    Executes a structured ReAct execution framework turn manually, tracking tool invocations,
    streaming delta events, and performing error reflection self-correction up to 5 steps deep.
    """
    history_store = get_session_history(session_id)
    raw_history = (
        await history_store.aget_messages()
        if hasattr(history_store, "aget_messages")
        else history_store.messages
    )
    trimmed_history = trimmer.invoke(raw_history)

    scratchpad = []
    max_iterations = 5

    for iteration in range(max_iterations):
        inputs = {
            "input": input_message,
            "chat_history": trimmed_history,
            "agent_scratchpad": scratchpad,
        }

        formatted_prompt = await prompt_template.ainvoke(inputs)
        ai_message = await llm.ainvoke(formatted_prompt)

        # Scrape and isolate internal thinking structures
        reasoning = ai_message.additional_kwargs.get("reasoning_content", "")
        if not reasoning and "<thought>" in ai_message.content:
            try:
                reasoning = (
                    ai_message.content.split("<thought>")[1]
                    .split("</thought>")[0]
                    .strip()
                )
            except Exception:
                pass

        if stream_queue and reasoning:
            await stream_queue.put({"type": "reasoning", "content": reasoning})

        # Append runtime execution footprints into the local tracking scratchpad
        scratchpad.append(ai_message)

        # Evaluate if the model wants to call tools
        if ai_message.tool_calls:
            for tool_call in ai_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                call_id = tool_call.get("id", f"call_{iteration}")

                if stream_queue:
                    await stream_queue.put(
                        {
                            "type": "tool_call",
                            "content": {"tool": tool_name, "input": tool_args},
                        }
                    )

                # Target core action processing with inline error-catch reflection architectures
                if tool_name in TOOLS_MAP:
                    try:
                        tool_output = await TOOLS_MAP[tool_name].ainvoke(tool_args)
                        obs_content = str(tool_output)
                    except Exception as tool_err:
                        # Wrap tool exceptions directly into localized feedback notifications for reflection
                        obs_content = f"Tool Execution Error: {str(tool_err)}. Re-analyze constraints, fix execution steps, and re-attempt."
                else:
                    obs_content = f"Error: Specified agent tool '{tool_name}' is currently offline/unavailable."

                if stream_queue:
                    await stream_queue.put(
                        {"type": "observation", "content": obs_content}
                    )

                scratchpad.append(
                    ToolMessage(content=obs_content, tool_call_id=call_id)
                )

            # Recirculate back up to next iteration loop sequence
            continue

        else:
            # Execution concluded safely. Clear wrapper markup blocks
            raw_output = ai_message.content
            if "<final_answer>" in raw_output:
                raw_output = (
                    raw_output.split("<final_answer>")[1]
                    .split("</final_answer>")[0]
                    .strip()
                )

            # Record clean chat state logs down to DB layer
            await history_store.aadd_messages(
                [HumanMessage(content=input_message), AIMessage(content=raw_output)]
            )
            return raw_output

    fallback_msg = "Agent timed out. Execution limits exceeded loop boundaries without clean state conclusions."
    await history_store.aadd_messages(
        [HumanMessage(content=input_message), AIMessage(content=fallback_msg)]
    )
    return fallback_msg


# ─── 4. FASTAPI APP ROUTER ENDPOINTS ────────────────────────────────────────
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
    try:
        raw_output = await run_react_loop(request.message, request.session_id)
        structured_data = parser.parse(raw_output)
        return {
            "data": structured_data,
            "session_id": request.session_id,
        }
    except Exception as e:
        error_msg = str(e).lower()
        if "rate_limit" in error_msg or "429" in error_msg:
            return {
                "data": {
                    "answer": "System rate limit hit. Please wait 60s.",
                    "files_modified": [],
                    "commands_run": [],
                    "confidence": 0,
                },
                "error": "rate_limit",
                "session_id": request.session_id,
            }
        return {
            "data": {
                "answer": raw_output if "raw_output" in locals() else str(e),
                "files_modified": [],
                "commands_run": [],
                "confidence": 0.5,
            },
            "error": f"Execution processing error: {str(e)}",
            "session_id": request.session_id,
        }


@app.post("/api/chat/stream")
async def chat_endpoint_stream(request: ChatRequest):
    stream_queue = asyncio.Queue()

    async def producer():
        try:
            final_ans = await run_react_loop(
                request.message, request.session_id, stream_queue=stream_queue
            )
            try:
                parsed_ans = parser.parse(final_ans).answer
            except Exception:
                parsed_ans = final_ans
            await stream_queue.put({"type": "answer", "content": parsed_ans})
        except Exception as e:
            await stream_queue.put({"type": "error", "content": str(e)})
        finally:
            await stream_queue.put({"type": "done"})

    async def event_generator():
        task = asyncio.create_task(producer())
        while True:
            item = await stream_queue.get()
            if item["type"] == "done":
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                break
            yield f"data: {json.dumps(item)}\n\n"
            stream_queue.task_done()
        await task

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
