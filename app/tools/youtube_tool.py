from pydantic import BaseModel, Field
from langchain_core.tools import tool
from typing import Dict, Any, List
from youtubesearchpython import VideosSearch


class YouTubeSearchSchema(BaseModel):
    query: str = Field(
        description="The search query keywords to find videos on YouTube (e.g. 'python tutorials for beginners')."
    )
    limit: int = Field(
        default=5,
        description="The maximum number of search results to retrieve (max 20).",
    )


@tool("search_youtube_videos", args_schema=YouTubeSearchSchema)
def search_youtube_videos(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Use this tool to search YouTube for videos matching a keyword query.
    Input must be a raw search query string and a limit integer.
    Do not attempt to answer the user directly when using this tool.
    """

    try:
        # Enforce the maximum limit of 20
        search_limit = min(limit, 20)

        # Initialize the VideosSearch object and instantly fetch results
        videos_search = VideosSearch(query, limit=search_limit)
        results = videos_search.result()

        # If results are empty or none, return a safe message
        if not results or "result" not in results:
            return {
                "success": True,
                "query": query,
                "results": [],
                "message": "No videos found.",
            }

        # Parse the structured response
        formatted_results = []
        for video in results.get("result", []):
            formatted_results.append(
                {
                    "title": video.get("title"),
                    "link": video.get("link"),
                    "duration": video.get("duration"),
                    "view_count": video.get("viewCount", {}).get(
                        "short", "N/A"
                    ),  # 'short' or 'text' works depending on version
                    "published_time": video.get("publishedTime"),
                    "channel_name": video.get("channel", {}).get("name"),
                }
            )

        return {"success": True, "query": query, "results": formatted_results}

    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch YouTube search results: {str(e)}",
        }
