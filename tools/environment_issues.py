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
# 6. SYSTEM / ENVIRONMENT  ← debug env issues
# ════════════════════════════════════════════════════════════════


@tool
@resilient_tool(max_retries=1)
def get_environment_info() -> dict:
    """Get system info: Python version, OS, current directory, PATH, env vars.
    Use to debug environment issues, check if packages are installed,
    or understand the runtime context.
    """
    import sys, platform

    result = subprocess.run(
        ["pip", "list", "--format=json"], capture_output=True, text=True
    )
    try:
        import json

        packages = {p["name"]: p["version"] for p in json.loads(result.stdout)}
    except Exception:
        packages = {}

    return {
        "python_version": sys.version,
        "platform": platform.system(),
        "cwd": os.getcwd(),
        "installed_packages_count": len(packages),
        "key_packages": {  # only show relevant ones
            k: v
            for k, v in packages.items()
            if k
            in {
                "langchain",
                "fastapi",
                "uvicorn",
                "langchain-groq",
                "langchain-core",
                "httpx",
                "sqlalchemy",
                "pydantic",
            }
        },
    }


@tool
@resilient_tool(max_retries=1)
def get_env_variable(var_name: str) -> str:
    """Read an environment variable by name. Returns its value or a not-found message.
    Use to check if API keys, database URLs, or config vars are set.
    Example: get_env_variable('DATABASE_URL')
    """
    value = os.getenv(var_name)
    if value is None:
        return f"[Not set: {var_name}]"
    # Mask secrets — show only first 4 chars
    if any(kw in var_name.upper() for kw in ["KEY", "SECRET", "TOKEN", "PASSWORD"]):
        return f"{value[:4]}{'*' * (len(value) - 4)} (masked)"
    return value


@tool
@resilient_tool(max_retries=1)
def check_disk_usage(path: str = ".") -> dict:
    """Check disk space usage at a path. Useful for large codebases or logs."""
    total, used, free = shutil.disk_usage(path)
    return {
        "total_gb": round(total / 1e9, 2),
        "used_gb": round(used / 1e9, 2),
        "free_gb": round(free / 1e9, 2),
        "used_percent": round(used / total * 100, 1),
    }
