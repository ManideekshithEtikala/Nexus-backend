import os
import json
import asyncio
import uuid
from contextvars import ContextVar
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
from core.config import settings

# ContextVars to store main loop and stream queue for the current request context
current_main_loop: ContextVar[asyncio.AbstractEventLoop] = ContextVar("current_main_loop")
current_stream_queue: ContextVar[asyncio.Queue] = ContextVar("current_stream_queue")

# Import tools for subagent bindings
from tools import (
    read_file, read_file_lines, write_file, append_to_file, list_files, delete_file, copy_file, move_file,
    run_command, run_python_code, get_running_processes, kill_process,
    search_in_files, str_replace_in_file, get_file_structure, count_lines_of_code,
    git_status, git_diff, git_log, git_commit,
    fetch_url, search_pypi, browse_web, capture_local_ui,
    get_environment_info, get_env_variable, check_disk_usage
)

# Central tools catalog mapping
TOOLS_MAP = {
    "read_file": read_file,
    "read_file_lines": read_file_lines,
    "write_file": write_file,
    "append_to_file": append_to_file,
    "list_files": list_files,
    "delete_file": delete_file,
    "copy_file": copy_file,
    "move_file": move_file,
    "run_command": run_command,
    "run_python_code": run_python_code,
    "get_running_processes": get_running_processes,
    "kill_process": kill_process,
    "search_in_files": search_in_files,
    "str_replace_in_file": str_replace_in_file,
    "get_file_structure": get_file_structure,
    "count_lines_of_code": count_lines_of_code,
    "git_status": git_status,
    "git_diff": git_diff,
    "git_log": git_log,
    "git_commit": git_commit,
    "fetch_url": fetch_url,
    "search_pypi": search_pypi,
    "browse_web": browse_web,
    "capture_local_ui": capture_local_ui,
    "get_environment_info": get_environment_info,
    "get_env_variable": get_env_variable,
    "check_disk_usage": check_disk_usage,
}


class AgentConfig:
    def __init__(self, name: str, description: str, system_prompt: str, tools: list[str], temperature: float = 0.2):
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.tools = tools
        self.temperature = temperature


class AgentManager:
    """Manages system-level micro-agent persona and tools configurations."""
    def __init__(self):
        self.registry: dict[str, AgentConfig] = {}
        self._register_default_agents()

    def _register_default_agents(self):
        # 1. Coder Subagent
        self.registry["coder"] = AgentConfig(
            name="coder",
            description="Specialized codebase writer and file system navigator.",
            system_prompt=(
                "You are the Nexus Coder Subagent. Your purpose is to explore directories, read source files, "
                "scaffold new files, search patterns, and edit code lines using exact line replacements. "
                "Always verify your changes. Explain your technical reasoning before editing."
            ),
            tools=["read_file", "read_file_lines", "write_file", "append_to_file", "list_files", "search_in_files", "str_replace_in_file", "get_file_structure", "count_lines_of_code"],
            temperature=0.2
        )

        # 2. Shell Subagent
        self.registry["shell"] = AgentConfig(
            name="shell",
            description="System terminal execution and runtime diagnostic utility.",
            system_prompt=(
                "You are the Nexus Shell/System Subagent. Your purpose is to run terminal commands, compile and run "
                "inline python scripts, inspect active OS processes, and retrieve environment variable shapes. "
                "Always print the complete standard output/error and explain the command outcome."
            ),
            tools=["run_command", "run_python_code", "get_running_processes", "kill_process", "get_environment_info", "get_env_variable", "check_disk_usage"],
            temperature=0.1
        )

        # 3. Git Subagent
        self.registry["git_agent"] = AgentConfig(
            name="git_agent",
            description="Version control audit and commit manager.",
            system_prompt=(
                "You are the Nexus Git Subagent. Your purpose is to check version control status, fetch logs, review diffs, "
                "and commit files. Always inspect git differences before triggering commits and write clean commit messages."
            ),
            tools=["git_status", "git_diff", "git_log", "git_commit"],
            temperature=0.2
        )

        # 4. Researcher Subagent
        self.registry["researcher"] = AgentConfig(
            name="researcher",
            description="Web scraping and technical documentation gatherer.",
            system_prompt=(
                "You are the Nexus Researcher Subagent. Your purpose is to scrape live web URLs and inspect PyPI package structures. "
                "Filter out unnecessary HTML boilerplate and synthesize clean, technical responses focusing on usage APIs."
            ),
            tools=["fetch_url", "search_pypi", "browse_web", "capture_local_ui", "list_files", "read_file"],
            temperature=0.3
        )

    def get_agent(self, name: str) -> AgentConfig:
        if name not in self.registry:
            raise ValueError(f"Agent with name '{name}' is not registered.")
        return self.registry[name]


# Global instance of AgentManager
agent_manager = AgentManager()


async def run_subagent(agent_name: str, task: str, stream_queue: asyncio.Queue = None) -> str:
    """
    Executes a specialized subagent in a standalone, read-eval ReAct loop.
    Returns the compiled final response of the subagent.
    """
    try:
        config = agent_manager.get_agent(agent_name)
        print(f"[Subagent-{agent_name}] Initializing with task: '{task}'")

        if stream_queue:
            await stream_queue.put({
                "type": "reasoning",
                "content": f"\n\n[Spinning up Subagent: {agent_name.upper()}...]\n"
            })

        # Initialize ChatGroq with subagent-specific credentials
        llm = ChatGroq(
            model=os.getenv("GROQ_MODEL", "qwen/qwen3-32b"),
            api_key=settings.API_KEY,
            temperature=config.temperature,
            max_tokens=1024
        )

        # Map tools list
        subagent_tools = [TOOLS_MAP[t] for t in config.tools if t in TOOLS_MAP]
        llm_with_tools = llm.bind_tools(subagent_tools)

        # Build initial prompt
        system_message = SystemMessage(
            content=config.system_prompt + "\n\nOperate strictly in this loop: Thought -> Action -> Observation.\n"
                    "You must output your reasoning inside <thought></thought> tags before calling any tool.\n"
                    "When done, output your final answer wrapped inside <final_answer></final_answer> tags."
        )
        task_message = HumanMessage(content=task)
        messages = [system_message, task_message]

        max_iterations = 4
        iteration = 0
        final_answer = ""

        while iteration < max_iterations:
            iteration += 1
            print(f"[Subagent-{agent_name}] Iteration {iteration} starting...")

            response_message = await llm_with_tools.ainvoke(messages)
            text_content = response_message.content if response_message.content else ""
            tool_calls = response_message.tool_calls if response_message.tool_calls else []

            # Stream intermediate thought states to main SSE queue if provided
            if stream_queue and text_content:
                # Extract clean thought blocks
                import re
                think_match = re.search(r"<thought>(.*?)</thought>", text_content, re.DOTALL)
                if think_match:
                    think_text = think_match.group(1).strip()
                else:
                    think_text = text_content.replace("<think>", "").replace("</think>", "").strip()
                
                if think_text:
                    await stream_queue.put({
                        "type": "reasoning",
                        "content": f"\n[{agent_name.upper()} thinking]: {think_text}\n"
                    })

            # Handlers for fallback text-based tool matches
            actual_tool_calls = tool_calls
            if not actual_tool_calls and text_content:
                import re
                action_match = re.search(r"Action:\s*(\w+)", text_content)
                action_input_match = re.search(r"Action Input:\s*(.+)", text_content)
                if action_match and action_input_match:
                    tool_name = action_match.group(1).strip()
                    tool_input_str = action_input_match.group(1).strip()
                    try:
                        tool_args = json.loads(tool_input_str)
                        if not isinstance(tool_args, dict):
                            tool_args = {"value": tool_args}
                    except Exception:
                        tool_args = {"value": tool_input_str.strip('"\'')}
                    
                    actual_tool_calls = [{
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"sub_call_{uuid.uuid4().hex}"
                    }]

            if actual_tool_calls:
                messages.append(response_message)
                for tc in actual_tool_calls:
                    if isinstance(tc, dict):
                        name = tc["name"]
                        args = tc["args"]
                        call_id = tc["id"]
                    else:
                        name = tc.get("name")
                        args = tc.get("args", {})
                        call_id = tc.get("id", f"sub_call_{uuid.uuid4().hex}")

                    print(f"[Subagent-{agent_name}] Executing tool '{name}' with {args}")
                    
                    if stream_queue:
                        await stream_queue.put({
                            "type": "tool_call",
                            "content": {"tool": f"{agent_name}:{name}", "input": str(args)}
                        })

                    # Execute tool in threadpool
                    if name in TOOLS_MAP:
                        try:
                            tool_instance = TOOLS_MAP[name]
                            observation = await asyncio.to_thread(tool_instance.invoke, args)
                        except Exception as e:
                            observation = f"[Subagent tool error: {e}]"
                    else:
                        observation = f"[Subagent error: Tool {name} not found]"

                    if stream_queue:
                        await stream_queue.put({
                            "type": "observation",
                            "content": f"[Result of {name}]: {str(observation)[:500]}"
                        })

                    messages.append(ToolMessage(content=str(observation), tool_call_id=call_id))
                    
                    # Self-Correction Interceptor Injection for Subagents
                    if "[Subagent error" in str(observation) or "[Subagent tool error" in str(observation):
                        error_hint = ""
                        if "FileNotFoundError" in str(observation):
                            error_hint = " Hint: verify the file structure by checking directory contents."
                        elif "PermissionError" in str(observation):
                            error_hint = " Hint: check path permissions."
                        
                        correction_prompt = (
                            f"ATTENTION: The tool '{name}' failed with execution error: {observation}.{error_hint} "
                            "Please explain the issue inside your next <thought> block, self-correct your strategy or arguments, "
                            "and trigger the correct tool to achieve the goal successfully."
                        )
                        messages.append(SystemMessage(content=correction_prompt))
                        print(f"[Subagent-{agent_name}] Self-Correction interceptor triggered for failed tool '{name}'.")
                continue
            else:
                final_answer = text_content
                break

        # Sanitize final tags
        if "<final_answer>" in final_answer:
            final_answer = final_answer.split("<final_answer>")[1].split("</final_answer>")[0].strip()
        else:
            final_answer = final_answer.replace("</final_answer>", "").replace("<final_answer>", "").strip()

        if stream_queue:
            await stream_queue.put({
                "type": "reasoning",
                "content": f"\n[Subagent {agent_name.upper()} task completed.]\n\n"
            })

        print(f"[Subagent-{agent_name}] Finished with answer: {final_answer[:100]}...")
        return final_answer

    except Exception as e:
        error_msg = f"[Failure running subagent {agent_name}: {e}]"
        print(error_msg)
        return error_msg
