import os
import dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional, Literal
from tavily import AsyncTavilyClient

dotenv.load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

# 1. Keep ONLY your input validation model (ensures Groq sends flat, safe parameters)
class UniversalWebSearchSchema(BaseModel):
    query: str = Field(description="The search query compressed into 2-4 strict keywords.")
    topic: Literal["general", "news", "finance", "shopping"] = "general"
    time_range: Literal["any", "day", "week", "month", "year"] = "any"
    max_results: int = 5
    restrict_to_site: Optional[str] = None

@tool(args_schema=UniversalWebSearchSchema)
async def web_search_tool(
    query: str,
    topic: Literal["general", "news", "finance", "shopping"] = "general",
    time_range: Literal["any", "day", "week", "month", "year"] = "any",
    max_results: int = 5,
    restrict_to_site: Optional[str] = None
) -> str: #  FORCE return hint to 'str'
    """
    Optimized web search engine capability spanning multiple indices.
    """
    tavily_topic = "general" if topic == "shopping" else topic
    tavily_time_range = None if time_range == "any" else time_range
    include_domains = [restrict_to_site] if restrict_to_site else None

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

    # 🎯 PRODUCTION ARCHITECTURE: Serialize everything into a unified text bridge block!
    return (
        f"=== WEB SEARCH TOOL EXECUTION REPORT ===\n"
        f"SEARCH_QUERY: {query}\n"
        f"SUMMARY_ANSWER: {raw_answer}\n\n"
        f"EXTRACTED_SOURCES:\n{sources_block}\n"
        f"========================================"
    )