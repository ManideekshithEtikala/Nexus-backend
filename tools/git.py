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
# 4. GIT TOOLS  ← essential for a coding agent
# ════════════════════════════════════════════════════════════════


@tool
@resilient_tool(max_retries=1)
def git_status(repo_path: str = ".") -> str:
    """Get the current git status of a repository.
    Shows modified, staged, and untracked files.
    """
    result = subprocess.run(
        ["git", "status"], cwd=repo_path, capture_output=True, text=True
    )
    return result.stdout or result.stderr


@tool
@resilient_tool(max_retries=1)
def git_diff(repo_path: str = ".", file_path: str = "") -> str:
    """Show git diff — what changed since last commit.
    Pass file_path to diff a specific file, or leave empty for all changes.
    """
    cmd = ["git", "diff"]
    if file_path:
        cmd.append(file_path)
    result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True)
    output = result.stdout
    # Cap diff output to avoid flooding context
    if len(output) > 3000:
        output = output[:3000] + "\n... [truncated — diff too large]"
    return output or "No changes."


@tool
@resilient_tool(max_retries=1)
def git_log(repo_path: str = ".", n: int = 10) -> str:
    """Get the last N git commits with messages and authors."""
    result = subprocess.run(
        ["git", "log", f"-{n}", "--oneline", "--decorate"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    return result.stdout or result.stderr


@tool
@resilient_tool(max_retries=1)
def git_commit(repo_path: str, message: str, add_all: bool = True) -> str:
    """Stage all changes and create a git commit.
    add_all=True stages everything (git add -A) before committing.
    """
    if add_all:
        subprocess.run(["git", "add", "-A"], cwd=repo_path)
    result = subprocess.run(
        ["git", "commit", "-m", message], cwd=repo_path, capture_output=True, text=True
    )
    return result.stdout or result.stderr
