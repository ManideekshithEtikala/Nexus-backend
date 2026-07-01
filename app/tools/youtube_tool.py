import os
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


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
    # 1. Grab API Key safely from environment variables
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        return {
            "success": False,
            "error": "YouTube API Key is missing. Please ensure YOUTUBE_API_KEY is set in your environment.",
        }

    try:
        # Enforce the maximum limit of 20
        search_limit = min(limit, 20)

        # 2. Initialize the official Google API Client
        youtube = build("youtube", "v3", developerKey=api_key)

        # 3. Request search from YouTube API v3
        request = youtube.search().list(
            q=query,
            part="snippet",
            type="video",  # Ensures we only get videos (no channels or playlists)
            maxResults=search_limit,
            videoEmbeddable="true",  # Great for clean rendering later
        )
        response = request.execute()
        # print(response)
        # 4. Parse the structured official response to match your existing schema
        formatted_results = []
        for item in response.get("items", []):
            video_id = item.get("id", {}).get("videoId")
            snippet = item.get("snippet", {})

            if not video_id:
                continue

            formatted_results.append(
                {
                    "title": snippet.get("title"),
                    "link": f"https://www.youtube.com/watch?v={video_id}",
                    # Note: YouTube search endpoint doesn't give duration or views directly
                    # without an extra API call, so we fallback gracefully.
                    "duration": "Available on watch",
                    "view_count": "N/A",
                    "published_time": snippet.get("publishedAt"),
                    "channel_name": snippet.get("channelTitle"),
                }
            )

        return {"success": True, "query": query, "results": formatted_results}

    except HttpError as e:
        return {
            "success": False,
            "error": f"YouTube API HTTP error occurred: {e.reason}",
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to fetch YouTube search results: {str(e)}",
        }
