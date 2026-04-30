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
# 3. CODE INTELLIGENCE  ← makes agent understand codebases
# ════════════════════════════════════════════════════════════════


@tool
@resilient_tool(max_retries=1)
def search_in_files(directory: str, query: str, file_pattern: str = "*.py") -> str:
    """Search for a string/pattern across all files in a directory (like grep).
    Returns matching lines with their file path and line number.
    Use to find where a function is defined, where a variable is used, etc.
    Example: search_in_files('.', 'def authenticate', '*.py')
    Example: search_in_files('./src', 'TODO', '*.ts')
    """
    results = []
    path = Path(directory)

    for filepath in path.rglob(file_pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if query.lower() in line.lower():
                        results.append(f"{filepath}:{line_num}: {line.rstrip()}")
        except (UnicodeDecodeError, PermissionError):
            continue

    if not results:
        return f"No matches found for '{query}' in {directory}/{file_pattern}"
    return "\n".join(results[:50])  # cap at 50 matches to avoid token blowup


@tool
@resilient_tool(max_retries=1)
def str_replace_in_file(file_path: str, old_str: str, new_str: str) -> str:
    """Replace an exact string in a file with a new string.
    Much safer than rewriting the whole file — only changes what needs changing.
    old_str must match exactly (including whitespace/indentation).
    Example: fix a specific function, rename a variable, update a config value.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if old_str not in content:
        return f"[Error: exact string not found in {file_path}. Check whitespace/indentation.]"

    count = content.count(old_str)
    if count > 1:
        return f"[Error: string appears {count} times. Make old_str more specific to match exactly once.]"

    new_content = content.replace(old_str, new_str, 1)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    return f"Replaced in {file_path} successfully."


@tool
@resilient_tool(max_retries=1)
def get_file_structure(directory: str, max_depth: int = 3) -> str:
    """Get a tree view of directory structure up to max_depth levels.
    Use at the start of a task to understand the project layout.
    Example: get_file_structure('./my_project', max_depth=2)
    """
    result = []

    def _walk(path: Path, depth: int, prefix: str = ""):
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda e: (e.is_file(), e.name))
        except PermissionError:
            return

        for i, entry in enumerate(entries):
            # Skip hidden and common noise dirs
            if entry.name.startswith(".") or entry.name in {
                "__pycache__",
                "node_modules",
                ".git",
                "venv",
                ".venv",
                "dist",
                "build",
            }:
                continue
            connector = "└── " if i == len(entries) - 1 else "├── "
            result.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                _walk(entry, depth + 1, prefix + extension)

    result.append(str(directory))
    _walk(Path(directory), 1)
    return "\n".join(result)


@tool
@resilient_tool(max_retries=1)
def count_lines_of_code(directory: str, file_pattern: str = "*.py") -> dict:
    """Count total lines of code across all matching files.
    Returns per-file counts and total. Useful for understanding project size.
    """
    file_counts = {}
    path = Path(directory)

    for filepath in path.rglob(file_pattern):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                non_empty = sum(
                    1 for l in lines if l.strip() and not l.strip().startswith("#")
                )
                file_counts[str(filepath)] = {"total": len(lines), "code": non_empty}
        except (UnicodeDecodeError, PermissionError):
            continue

    total_code = sum(v["code"] for v in file_counts.values())
    return {"files": file_counts, "total_code_lines": total_code}
