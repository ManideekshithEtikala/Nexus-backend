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
# 5. WEB / DOCS SEARCH  ← lets agent look things up
# ════════════════════════════════════════════════════════════════


@tool
@resilient_tool(max_retries=1)
def fetch_url(url: str) -> str:
    """Fetch the text content of any URL — documentation, GitHub files, APIs.
    Use to read official docs, fetch a raw GitHub file, or check an endpoint.
    Example: fetch_url('https://raw.githubusercontent.com/user/repo/main/README.md')
    Example: fetch_url('https://docs.python.org/3/library/asyncio.html')
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; CodingAgent/1.0)"}
        response = httpx.get(url, headers=headers, timeout=10, follow_redirects=True)
        response.raise_for_status()
        text = response.text
        # Strip HTML tags roughly for readability
        import re

        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Cap to avoid flooding context window
        return text[:4000] + "... [truncated]" if len(text) > 4000 else text
    except Exception as e:
        return f"[Error fetching {url}: {str(e)}]"


@tool
@resilient_tool(max_retries=1)
def search_pypi(package_name: str) -> dict:
    """Look up a Python package on PyPI — version, description, homepage.
    Use before installing to verify package name and get info.
    Example: search_pypi('langchain-groq')
    """
    try:
        response = httpx.get(f"https://pypi.org/pypi/{package_name}/json", timeout=8)
        response.raise_for_status()
        data = response.json()["info"]
        return {
            "name": data["name"],
            "version": data["version"],
            "summary": data["summary"],
            "homepage": data["home_page"] or data.get("project_url", ""),
            "requires_python": data["requires_python"],
            "license": data["license"],
        }
    except Exception as e:
        return {"error": f"Package not found or API error: {str(e)}"}
