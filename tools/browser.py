import os
import re
import uuid
from playwright.sync_api import sync_playwright
from markdownify import markdownify as md
from langchain_core.tools import tool
from .error_handling import resilient_tool

@tool
@resilient_tool(max_retries=1)
def browse_web(url: str, action: str = "scrape") -> str:
    """Launch a headless browser using Playwright to browse/scrape a webpage and convert it to clean markdown.
    Args:
        url: The absolute HTTP/HTTPS URL to browse.
        action: The browse action to perform, e.g., 'scrape' or 'extract'. Default is 'scrape'.
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Extract content
            html = page.content()
            browser.close()
            
            # Convert to markdown
            markdown_content = md(html, strip=['script', 'style', 'noscript', 'iframe'])
            
            # Clean up excessive whitespace
            markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)
            markdown_content = re.sub(r' +', ' ', markdown_content)
            
            # Cap to avoid context overflow
            if len(markdown_content) > 5000:
                return markdown_content[:5000] + "\n\n... [Content truncated due to length]"
            return markdown_content
    except Exception as e:
        return f"[Error browsing {url}: {str(e)}]"


@tool
@resilient_tool(max_retries=1)
def capture_local_ui(url: str) -> str:
    """Capture a screenshot of a local or web URL using Playwright and save it under storage/screenshots/
    Args:
        url: The URL of the UI to capture (e.g., http://localhost:3000).
    """
    try:
        # Determine storage path inside workspace
        workspace_dir = "/Users/manideekshith/Desktop/nvidia"
        screenshots_dir = os.path.join(workspace_dir, "storage", "screenshots")
        os.makedirs(screenshots_dir, exist_ok=True)
        
        filename = f"screenshot_{uuid.uuid4().hex}.png"
        filepath = os.path.join(screenshots_dir, filename)
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={"width": 1280, "height": 800}
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.screenshot(path=filepath, full_page=True)
            browser.close()
            
        return f"[Screenshot successfully captured and saved to: {filepath}]"
    except Exception as e:
        return f"[Error capturing UI at {url}: {str(e)}]"
