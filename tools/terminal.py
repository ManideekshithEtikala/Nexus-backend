# tools.py
import os
import subprocess
import shutil
import fnmatch
import httpx
from pathlib import Path
from langchain_core.tools import tool
from .error_handling import resilient_tool

# ════════════════════════════════════════════════════════════════
# 2. TERMINAL / SHELL  ← most important missing category
# ════════════════════════════════════════════════════════════════


@tool
@resilient_tool(max_retries=1)
def run_command(command: str, working_dir: str = ".") -> dict:
    """Run any shell command and return stdout, stderr, and exit code.
    Use for: running Python scripts, installing packages, running tests,
    starting/stopping servers, git commands, npm/pip commands.
    Example: run_command('python main.py')
    Example: run_command('pip install requests')
    Example: run_command('pytest tests/', working_dir='./my_project')
    """
    result = subprocess.run(
        command,
        shell=True,
        cwd=working_dir,
        capture_output=True,
        text=True,
        timeout=60,  # 60s timeout — prevents runaway processes
    )
    return {
        "stdout": result.stdout.strip() or "(no output)",
        "stderr": result.stderr.strip() or "(no errors)",
        "exit_code": result.returncode,
        "success": result.returncode == 0,
    }


@tool
@resilient_tool(max_retries=1)
def run_python_code(code: str) -> dict:
    """Execute a Python code snippet directly and return the output.
    Use this to test small pieces of code, run calculations, or
    validate logic without creating a file first.
    Example: run_python_code('print([x**2 for x in range(10)])')
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    result = subprocess.run(
        ["python", tmp_path], capture_output=True, text=True, timeout=30
    )
    os.unlink(tmp_path)  # clean up temp file

    return {
        "stdout": result.stdout.strip() or "(no output)",
        "stderr": result.stderr.strip() or "(no errors)",
        "exit_code": result.returncode,
        "success": result.returncode == 0,
    }


@tool
@resilient_tool(max_retries=1)
def get_running_processes() -> str:
    """List currently running processes. Useful for checking if a server is running."""
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    # Return only non-system processes (trim noise)
    lines = result.stdout.split("\n")
    return "\n".join(lines[:30])  # cap output


@tool
@resilient_tool(max_retries=1)
def kill_process(process_name: str) -> str:
    """Kill a process by name. Example: kill_process('uvicorn')"""
    result = subprocess.run(
        ["pkill", "-f", process_name], capture_output=True, text=True
    )
    if result.returncode == 0:
        return f"Killed processes matching: {process_name}"
    return f"No process found matching: {process_name}"
