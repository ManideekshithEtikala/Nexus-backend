import traceback
from langchain_core.messages import ToolMessage
from app.tools.Web_search import web_search_tool
from app.tools.Research_docs import research_docs_tool

# Centralized array tracking all tools
ALL_TOOLS = [
    web_search_tool,
    research_docs_tool,
]

# Automated dynamic map binding tool name -> tool instance
TOOL_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}

async def safe_execute_tool(tool_function, tool_input: dict, tool_call_id: str) -> ToolMessage:
    """
    Executes an agent tool securely, capturing any data or failure as an 
    immutable entry in the chat history instead of crashing.
    """
    try:
        # Run tool function logic asynchronously
        result = await tool_function.ainvoke(tool_input)
        
        return ToolMessage(
            content=str(result), #  Ensures tool response data is strictly written as plain string context
            tool_call_id=tool_call_id,
            status="success"
        )
    except Exception as e:
        error_context = f"Tool Execution Failure: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_context)  
        
        return ToolMessage(
            content=f"ERROR: Tool failed to execute. Reason: {str(e)}. Please attempt a different approach or tool fallback.",
            tool_call_id=tool_call_id,
            status="error"
        )
    """
    Executes an agent tool securely, capturing any failure as an 
    immutable entry in the chat history instead of crashing.
    """
    try:
        # Asynchronously run tool execution instance
        result = await tool_function.ainvoke(tool_input)
        
        return ToolMessage(
            content=str(result),
            tool_call_id=tool_call_id,
            status="success"
        )
    except Exception as e:
        # State Immutability: System Firewall logs context server-side
        error_context = f"Tool Execution Failure: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_context)  
        
        # Returns a clean message containing error payloads back to LLM context
        return ToolMessage(
            content=f"ERROR: Tool failed to execute. Reason: {str(e)}. Please attempt a different approach or tool fallback.",
            tool_call_id=tool_call_id,
            status="error"
        )