
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from typing import Any, Dict

@tool
async def research_docs_tool(query: str) -> str:
    """
    A tool to perform research on a given query and return summarized insights along with references.
    Returns:
    - str: sample test
    """
    return f"Research tool completed and it is working so far.you can say that the reasearch tool is sucesfully working and it is giving the expected output for the query: {query}"