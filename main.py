# Load environment variables FIRST, before importing anything else
from dotenv import load_dotenv
load_dotenv()

import os
import json
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# LangChain Imports
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import trim_messages, AIMessage, HumanMessage, ToolMessage, SystemMessage

# Local Imports
from prompts.prompt import NEXUS_SYSTEM_PROMPT, REACT_INJECTION
from models import parser
from database.database import async_engine, initialize_database_schemas
from models.schemas import ChatRequest
from memory.hierarchical_history import HierarchicalSQLChatMessageHistory
from memory.vector_service import VectorMemoryService
from memory.facts_extractor import FactsExtractor

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

app = FastAPI(title="Nexus Core Framework API", version="1.0.0")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    await initialize_database_schemas()

# Global instances of vector and extraction services
vector_memory = VectorMemoryService()
facts_extractor = FactsExtractor()


class RealTimeStreamParser:
    """
    Stateful real-time parser that scans streaming LLM token chunks.
    Separates content into reasoning events (<think>) and final answer blocks (<final_answer>)
    so the raw XML/JSON wrappers don't clutter the frontend conversation.
    """
    def __init__(self, stream_queue: asyncio.Queue):
        self.stream_queue = stream_queue
        self.buffer = ""
        self.in_think = False
        self.in_final = False
        self.accumulated_think = ""
        self.accumulated_final = ""

    async def process_token(self, token: str):
        self.buffer += token

        while True:
            if not self.in_think and not self.in_final:
                # Look for tags
                think_idx = self.buffer.find("<think>")
                final_idx = self.buffer.find("<final_answer>")

                if think_idx != -1 and (final_idx == -1 or think_idx < final_idx):
                    # Flush pre-tag text
                    pre_text = self.buffer[:think_idx]
                    if pre_text:
                        await self.stream_queue.put({"type": "token", "content": pre_text})
                    self.in_think = True
                    self.buffer = self.buffer[think_idx + 7:]
                    continue
                elif final_idx != -1:
                    # Flush pre-tag text
                    pre_text = self.buffer[:final_idx]
                    if pre_text:
                        await self.stream_queue.put({"type": "token", "content": pre_text})
                    self.in_final = True
                    self.buffer = self.buffer[final_idx + 14:]
                    continue
                else:
                    # Avoid outputting partial tags at the tail of the buffer
                    potential_tags = [
                        "<", "<t", "<th", "<thi", "<thin", "<think",
                        "<f", "<fi", "<fin", "<fina", "<final", "<final_",
                        "<final_a", "<final_an", "<final_ans", "<final_answ",
                        "<final_answe", "<final_answer"
                    ]
                    longest_partial = 0
                    for pt in potential_tags:
                        if self.buffer.endswith(pt):
                            longest_partial = len(pt)
                            break
                    
                    if longest_partial > 0:
                        output_len = len(self.buffer) - longest_partial
                        text_to_output = self.buffer[:output_len]
                        if text_to_output:
                            await self.stream_queue.put({"type": "token", "content": text_to_output})
                        self.buffer = self.buffer[output_len:]
                    else:
                        await self.stream_queue.put({"type": "token", "content": self.buffer})
                        self.buffer = ""
                    break

            elif self.in_think:
                end_think_idx = self.buffer.find("</think>")
                if end_think_idx != -1:
                    think_content = self.buffer[:end_think_idx]
                    self.accumulated_think += think_content
                    await self.stream_queue.put({"type": "reasoning", "content": think_content})
                    self.in_think = False
                    self.buffer = self.buffer[end_think_idx + 8:]
                    continue
                else:
                    potential_ends = ["<", "</", "</t", "</th", "</thi", "</thin", "</think"]
                    longest_partial = 0
                    for pe in potential_ends:
                        if self.buffer.endswith(pe):
                            longest_partial = len(pe)
                            break
                    
                    if longest_partial > 0:
                        output_len = len(self.buffer) - longest_partial
                        text_to_output = self.buffer[:output_len]
                        if text_to_output:
                            self.accumulated_think += text_to_output
                            await self.stream_queue.put({"type": "reasoning", "content": text_to_output})
                        self.buffer = self.buffer[output_len:]
                    else:
                        self.accumulated_think += self.buffer
                        await self.stream_queue.put({"type": "reasoning", "content": self.buffer})
                        self.buffer = ""
                    break

            elif self.in_final:
                end_final_idx = self.buffer.find("</final_answer>")
                if end_final_idx != -1:
                    final_content = self.buffer[:end_final_idx]
                    self.accumulated_final += final_content
                    self.in_final = False
                    self.buffer = self.buffer[end_final_idx + 15:]
                    continue
                else:
                    potential_ends = [
                        "<", "</", "</f", "</fi", "</fin", "</fina", "</final",
                        "</final_", "</final_a", "</final_an", "</final_ans",
                        "</final_answ", "</final_answe", "</final_answer"
                    ]
                    longest_partial = 0
                    for pe in potential_ends:
                        if self.buffer.endswith(pe):
                            longest_partial = len(pe)
                            break
                    
                    if longest_partial > 0:
                        output_len = len(self.buffer) - longest_partial
                        text_to_accumulate = self.buffer[:output_len]
                        if text_to_accumulate:
                            self.accumulated_final += text_to_accumulate
                        self.buffer = self.buffer[output_len:]
                    else:
                        self.accumulated_final += self.buffer
                        self.buffer = ""
                    break

    async def flush(self):
        if self.buffer:
            if self.in_think:
                self.accumulated_think += self.buffer
                await self.stream_queue.put({"type": "reasoning", "content": self.buffer})
            elif self.in_final:
                self.accumulated_final += self.buffer
            else:
                await self.stream_queue.put({"type": "token", "content": self.buffer})
            self.buffer = ""


async def update_long_term_memory_task(
    session_id: str, user_message: str, assistant_response: str
):
    """
    Background worker that runs asynchronously after a response turn is served.
    Extracts declarative metadata facts and tracks updates inside the Pinecone vector index.
    """
    try:
        print(
            f"[MemoryWorker] Beginning automated background memory extraction for session: {session_id}"
        )

        # Extract new structured elements from chat turn using Qwen via Groq
        extracted_facts = await facts_extractor.extract_new_facts(
            user_msg=user_message, assistant_ans=assistant_response
        )

        if extracted_facts:
            print(
                f"[MemoryWorker] Identified {len(extracted_facts)} new architectural or preference updates."
            )
            for fact in extracted_facts:
                # Synchronize to cloud vector db index synchronously (handles non-async safely inside wrapper)
                await vector_memory.upsert_fact(
                    fact_content=fact,
                    source_session_id=session_id,
                    user_id="default_user",
                )
        else:
            print(
                "[MemoryWorker] No notable or long-term structural changes identified in this session turn."
            )

    except Exception as e:
        print(
            f"[MemoryWorker] Critical failure running background thread synchronization task: {e}"
        )


@app.post("/api/chat/stream")
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint for Nexus, feeding real-time context injections
    from Pinecone down through Groq API endpoints.
    """
    stream_queue = asyncio.Queue()

    async def producer():
        try:
            # 1. Look up long-term history matches from our cloud-based text-embedding-004 vector index
            historical_context = await vector_memory.query_relevant_context(
                query=request.message, user_id="default_user", limit=4
            )

            # 2. Re-combine historical preferences context directly with system settings context
            combined_system_prompt = NEXUS_SYSTEM_PROMPT
            if historical_context:
                combined_system_prompt = (
                    f"{NEXUS_SYSTEM_PROMPT}\n\n{historical_context}"
                )

            # 3. Create the list of messages for the LLM from PostgreSQL database
            history = HierarchicalSQLChatMessageHistory(
                session_id=request.session_id,
                connection=async_engine,
                table_name="message_store",
                async_mode=True,
            )
            history_messages = await history.aget_messages()
            
            # Sanitize and normalize history messages to standard LangChain types, stripping internal metadata and thought blocks
            clean_history = []
            for msg in history_messages:
                content = msg.content
                if not content:
                    continue
                # Strip reasoning / thoughts to prevent bloating token context length
                if "<think>" in content and "</think>" in content:
                    parts = content.split("</think>")
                    content = parts[-1].strip()
                elif "<think>" in content:
                    content = content.split("<think>")[0].strip()
                
                # Strip final_answer tag formatting if present to keep history clean
                if "<final_answer>" in content and "</final_answer>" in content:
                    content = content.split("<final_answer>")[1].split("</final_answer>")[0].strip()
                
                if msg.type == "human":
                    clean_history.append(HumanMessage(content=content))
                elif msg.type in ("ai", "assistant"):
                    clean_history.append(AIMessage(content=content))
            
            system_message = SystemMessage(content=combined_system_prompt + "\n\n" + REACT_INJECTION)
            user_message = HumanMessage(content=request.message)
            
            messages = [system_message] + clean_history + [user_message]
            
            # Prepare tools list and map
            tools_list = [
                read_file, read_file_lines, write_file, append_to_file, list_files, delete_file, copy_file, move_file,
                run_command, run_python_code, get_running_processes, kill_process,
                search_in_files, str_replace_in_file, get_file_structure, count_lines_of_code,
                git_status, git_diff, git_log, git_commit,
                fetch_url, search_pypi,
                get_environment_info, get_env_variable, check_disk_usage
            ]
            tools_map = {t.name: t for t in tools_list}
            
            llm = ChatGroq(
                model="qwen/qwen3-32b",
                api_key=os.getenv("API_KEY"),
                temperature=0.3,
                max_tokens=1024  # Limit response size to prevent exceeding TPM on Groq
            )
            
            # Bound tools: Optimize prompt context size by only binding the essential developer tools.
            # This prevents rate limits and context-length issues on Groq's on-demand TPM limits.
            essential_tools = [
                read_file, write_file, list_files, run_command,
                search_in_files, str_replace_in_file, get_file_structure
            ]
            
            # Bind optimized tools list to the LLM
            llm_with_tools = llm.bind_tools(essential_tools)
            
            # Streaming parser helper
            parser_helper = RealTimeStreamParser(stream_queue)
            
            max_iterations = 6
            iteration = 0
            final_ans = ""
            
            print(f"[ChatEndpoint] Total messages to Qwen: {len(messages)}")
            for idx, msg in enumerate(messages):
                print(f"  Msg {idx}: {type(msg).__name__} | {len(msg.content)} characters")

            while iteration < max_iterations:
                iteration += 1
                print(f"[AgentLoop] Iteration {iteration} starting...")
                
                response_message = None
                
                # Resilient invocation wrapper with exponential backoff retries for Groq API limits
                max_retries = 3
                delay = 2.0
                for attempt in range(max_retries):
                    try:
                        response_message = await llm_with_tools.ainvoke(messages)
                        break  # Success!
                    except Exception as e:
                        if attempt < max_retries - 1 and ("rate_limit" in str(e).lower() or "429" in str(e) or "400" in str(e) or "reduce the length" in str(e).lower()):
                            print(f"[AgentLoop] Rate limit or token threshold hit. Retrying in {delay}s...")
                            await asyncio.sleep(delay)
                            delay *= 2
                        else:
                            raise e
                
                text_content = response_message.content if response_message.content else ""
                tool_calls = response_message.tool_calls if response_message.tool_calls else []
                
                # Stream only the thoughts/reasoning to the frontend during the agent turns
                if text_content and "<think>" in text_content:
                    import re
                    think_match = re.search(r"<think>(.*?)</think>", text_content, re.DOTALL)
                    if think_match:
                        think_text = think_match.group(1).strip()
                    else:
                        parts = text_content.split("<think>")
                        think_text = parts[-1].strip()
                    
                    if think_text:
                        chunk_size = 16
                        for idx in range(0, len(think_text), chunk_size):
                            chunk = think_text[idx:idx+chunk_size]
                            await stream_queue.put({"type": "reasoning", "content": chunk})
                            await asyncio.sleep(0.005)
                
                # Check for tool calls (either native or parsed from text)
                actual_tool_calls = tool_calls if tool_calls else (response_message.tool_calls if response_message else [])
                
                # Text-based fallback parsing for tool calls if native tool calls are empty
                if not actual_tool_calls and text_content:
                    import re
                    import uuid
                    action_match = re.search(r"Action:\s*(\w+)", text_content)
                    action_input_match = re.search(r"Action Input:\s*(.+)", text_content)
                    if action_match and action_input_match:
                        tool_name = action_match.group(1).strip()
                        tool_input_str = action_input_match.group(1).strip()
                        
                        try:
                            import json
                            tool_args = json.loads(tool_input_str)
                            if not isinstance(tool_args, dict):
                                tool_args = {"value": tool_args}
                        except Exception:
                            tool_args = {"value": tool_input_str.strip('"\'')}
                            
                        actual_tool_calls = [{
                            "name": tool_name,
                            "args": tool_args,
                            "id": f"call_fallback_{uuid.uuid4().hex}"
                        }]
                
                if actual_tool_calls:
                    print(f"[AgentLoop] LLM requested {len(actual_tool_calls)} tool calls.")
                    
                    # Append assistant's thoughts to message chain
                    messages.append(response_message)
                    
                    for tc in actual_tool_calls:
                        # Extract parameters (handles both standard ToolCall objects and dicts)
                        if isinstance(tc, dict):
                            name = tc["name"]
                            args = tc["args"]
                            call_id = tc["id"]
                        else:
                            name = tc.get("name")
                            args = tc.get("args", {})
                            call_id = tc.get("id", f"call_{uuid.uuid4().hex}")
                        
                        print(f"[AgentLoop] Running tool '{name}' with args {args}")
                        
                        # Stream tool call start to frontend
                        await stream_queue.put({
                            "type": "tool_call",
                            "content": {"tool": name, "input": str(args)}
                        })
                        
                        # Execute tool in a threadpool so it doesn't block the async loop
                        if name in tools_map:
                            try:
                                tool_instance = tools_map[name]
                                observation = await asyncio.to_thread(tool_instance.invoke, args)
                            except Exception as e:
                                observation = f"[Error executing tool {name}: {e}]"
                        else:
                            observation = f"[Error: Tool {name} not found]"
                            
                        print(f"[AgentLoop] Tool '{name}' observation result (truncated): {str(observation)[:100]}...")
                        
                        # Stream observation back to frontend
                        await stream_queue.put({
                            "type": "observation",
                            "content": str(observation)
                        })
                        
                        # Add tool response to message history
                        messages.append(ToolMessage(content=str(observation), tool_call_id=call_id))
                        
                    # Loop back to let model process observations
                    continue
                else:
                    # No tool calls; LLM final answer completed
                    final_ans = text_content
                    break
            
            # Clean final answer text
            final_raw = parser_helper.accumulated_final or final_ans
            try:
                parsed_ans = parser.parse(final_raw).answer
            except Exception:
                if "<final_answer>" in final_raw:
                    parsed_ans = final_raw.split("<final_answer>")[1].split("</final_answer>")[0].strip()
                else:
                    parsed_ans = final_raw

            # Stream the clean final answer to the user as clean token events
            if parsed_ans:
                chunk_size = 12
                for idx in range(0, len(parsed_ans), chunk_size):
                    chunk = parsed_ans[idx:idx+chunk_size]
                    await stream_queue.put({"type": "token", "content": chunk})
                    await asyncio.sleep(0.015)  # Premium typewriter micro-delay
            
            # Save transaction to permanent history
            await history.aadd_messages([
                HumanMessage(content=request.message),
                AIMessage(content=parsed_ans)
            ])

            # Spawn off background worker for long-term facts updates
            asyncio.create_task(
                update_long_term_memory_task(
                    session_id=request.session_id,
                    user_message=request.message,
                    assistant_response=parsed_ans,
                )
            )

        except Exception as e:
            print(f"[ChatEndpoint] Critical streaming runtime exception: {e}")
            msg_details = [(type(m).__name__, len(m.content)) for m in messages] if 'messages' in locals() else []
            await stream_queue.put({"type": "error", "content": f"{str(e)} | Messages: {msg_details}"})
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
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
