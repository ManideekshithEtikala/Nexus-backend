import os
import dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Literal
from tavily import AsyncTavilyClient

dotenv.load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

# 1. Add "" to the Literal options to satisfy Groq's edge validator
class UniversalWebSearchSchema(BaseModel):
    query: str = Field(description="The search query compressed into 2-4 strict keywords.")
    topic: Literal["general", "news", "finance", "shopping", ""] = "general"
    time_range: Literal["any", "day", "week", "month", "year"] = "any"
    max_results: int = 5
    restrict_to_site: Optional[str] = None

@tool(args_schema=UniversalWebSearchSchema)
async def web_search_tool(
    query: str,
    topic: Literal["general", "news", "finance", "shopping", ""] = "general",
    time_range: Literal["any", "day", "week", "month", "year"] = "any",
    max_results: int = 5,
    restrict_to_site: Optional[str] = None
) -> str:
    """
    Optimized web search engine capability spanning multiple indices.
    """
    # 🎯 DEFENSIVE GUARD: If the LLM sends an empty query string, fail gracefully instead of crashing Tavily
    if not query or not query.strip():
        return (
            f"=== WEB SEARCH TOOL EXECUTION REPORT ===\n"
            f"STATUS: FAILED\n"
            f"REASON: The model attempted a search call but provided an empty query string.\n"
            f"========================================"
        )

    # Clean up empty strings or shopping overrides safely inside Python
    tavily_topic = "general" if topic in ("", "shopping") else topic
    tavily_time_range = None if time_range == "any" else time_range
    include_domains = [restrict_to_site] if restrict_to_site else None

    try:
        # Run the network execution
        response = await tavily_client.search(
            query=query,
            topic=tavily_topic,
            time_range=tavily_time_range,
            max_results=max_results,
            include_domains=include_domains,
            include_answer=True,
            search_depth="advanced",
        )

        raw_answer = response.get("answer", "No direct automated text summary available.")
        
        # Extract URLs into a clean, identifiable section
        urls = [r.get("url") for r in response.get("results", []) if r.get("url")]
        unique_sources = list(set(urls))
        sources_block = "\n".join(f"- {url}" for url in unique_sources) or "No sources found."

        return (
            f"=== WEB SEARCH TOOL EXECUTION REPORT ===\n"
            f"SEARCH_QUERY: {query}\n"
            f"SUMMARY_ANSWER: {raw_answer}\n\n"
            f"EXTRACTED_SOURCES:\n{sources_block}\n"
            f"========================================"
        )
    except Exception as e:
        return f"=== WEB SEARCH TOOL EXECUTION REPORT ===\nERROR executing Tavily Search: {str(e)}\n========================================"