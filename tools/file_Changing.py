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
# 1. FILE I/O  (you already have these — upgraded versions)
# ════════════════════════════════════════════════════════════════


@tool
@resilient_tool(max_retries=1)
def read_file(file_path: str) -> str:
    """Read the full content of any file. Returns content as string."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        return f"[Binary file — cannot read as text: {file_path}]"
    except FileNotFoundError:
        return f"[Error: file not found: {file_path}]"


@tool
@resilient_tool(max_retries=1)
def read_file_lines(file_path: str, start_line: int, end_line: int) -> str:
    """Read specific line range from a file (1-indexed).
    Use this instead of read_file when you only need part of a large file.
    Example: read_file_lines('main.py', 10, 40)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        selected = lines[start_line - 1 : end_line]
        numbered = [f"{start_line + i}: {l}" for i, l in enumerate(selected)]
        return "".join(numbered)
    except FileNotFoundError:
        return f"[Error: file not found: {file_path}]"


@tool
@resilient_tool(max_retries=1)
def write_file(file_path: str, content: str) -> str:
    """Write (overwrite) content to a file. Creates parent directories if needed."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written: {file_path} ({len(content)} chars)"


@tool
@resilient_tool(max_retries=1)
def append_to_file(file_path: str, content: str) -> str:
    """Append content to end of a file without overwriting existing content."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(content)
    return f"Appended to {file_path}"


@tool
@resilient_tool(max_retries=1)
def list_files(directory: str, pattern: str = "*") -> list[str]:
    """List files in a directory. Optionally filter by pattern like '*.py' or '*.json'.
    Example: list_files('.', '*.py')
    """
    path = Path(directory)
    if not path.exists():
        return [f"[Error: directory not found: {directory}]"]
    
    matches = []
    ignore_dirs = {"node_modules", "venv", ".venv", ".git", "__pycache__", ".next", "dist", "build"}
    
    for p in path.rglob(pattern):
        # Ignore dependency folders and common build environments
        if any(segment in ignore_dirs for segment in p.parts):
            continue
        if p.is_file():
            matches.append(str(p))
            
    if len(matches) > 300:
        return sorted(matches[:300]) + [f"... and {len(matches) - 300} more files (truncated to avoid token context limits)"]
        
    return sorted(matches)


@tool
@resilient_tool(max_retries=1)
def delete_file(file_path: str) -> str:
    """Delete a file. Use with caution."""
    try:
        os.remove(file_path)
        return f"Deleted: {file_path}"
    except FileNotFoundError:
        return f"[Error: file not found: {file_path}]"


@tool
@resilient_tool(max_retries=1)
def copy_file(source: str, destination: str) -> str:
    """Copy a file from source to destination path."""
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return f"Copied {source} → {destination}"


@tool
@resilient_tool(max_retries=1)
def move_file(source: str, destination: str) -> str:
    """Move or rename a file."""
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(source, destination)
    return f"Moved {source} → {destination}"
