import asyncio
from langchain_core.tools import tool
from .error_handling import resilient_tool

@tool
@resilient_tool(max_retries=1)
def delegate_to_coder(task: str) -> str:
    """Delegate a codebase writing, editing, or navigation task to the expert Coder Subagent.
    Provide a specific, descriptive instruction of what files to read/modify/search.
    Example: delegate_to_coder('Read database/database.py and verify standard connections.')
    """
    try:
        from core.agents import run_subagent, current_main_loop, current_stream_queue
        loop = current_main_loop.get()
      
        queue = current_stream_queue.get()
        # Schedule the async subagent run in the main event loop and wait for result
        future = asyncio.run_coroutine_threadsafe(
            run_subagent("coder", task, queue),
            loop
        )
        return future.result()
    except Exception as e:
        return f"[Error delegating to coder subagent: {e}]"


@tool
@resilient_tool(max_retries=1)
def delegate_to_shell(task: str) -> str:
    """Delegate terminal commands, inline script execution, process tree checking, or environment diagnostics to the expert Shell Subagent.
    Provide precise instruction of what commands to execute or check.
    Example: delegate_to_shell('Check environment variables and verify PYTHONPATH.')
    """
    try:
        from core.agents import run_subagent, current_main_loop, current_stream_queue
        loop = current_main_loop.get()
        queue = current_stream_queue.get()
        future = asyncio.run_coroutine_threadsafe(
            run_subagent("shell", task, queue),
            loop
        )
        return future.result()
    except Exception as e:
        return f"[Error delegating to shell subagent: {e}]"


@tool
@resilient_tool(max_retries=1)
def delegate_to_git(task: str) -> str:
    """Delegate version control tasks like checking status, reviewing branch diffs, or committing staged modifications to the expert Git Subagent.
    Example: delegate_to_git('Check git status and explain untracked files.')
    """
    try:
        from core.agents import run_subagent, current_main_loop, current_stream_queue
        loop = current_main_loop.get()
        queue = current_stream_queue.get()
        future = asyncio.run_coroutine_threadsafe(
            run_subagent("git_agent", task, queue),
            loop
        )
        return future.result()
    except Exception as e:
        return f"[Error delegating to git subagent: {e}]"


@tool
@resilient_tool(max_retries=1)
def delegate_to_researcher(task: str) -> str:
    """Delegate web scraping, technical documentation gathering, or PyPI package lookup to the expert Researcher Subagent.
    Example: delegate_to_researcher('Scrape PyPI to find the latest version and description of fastapi.')
    """
    try:
        from core.agents import run_subagent, current_main_loop, current_stream_queue
        loop = current_main_loop.get()
        queue = current_stream_queue.get()
        future = asyncio.run_coroutine_threadsafe(
            run_subagent("researcher", task, queue),
            loop
        )
        return future.result()
    except Exception as e:
        return f"[Error delegating to researcher subagent: {e}]"
