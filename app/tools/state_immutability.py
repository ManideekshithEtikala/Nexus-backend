import json
import traceback
from pydantic import ValidationError
from langchain_core.messages import ToolMessage

from app.tools.Web_search import web_search_tool
from app.tools.Research_docs import research_docs_tool
from app.tools.change_behaviour_tool import change_behaviour_profile

# Centralized array tracking all tools
ALL_TOOLS = [
    web_search_tool,
    research_docs_tool,
    change_behaviour_profile
]

# Automated dynamic map binding tool name -> tool instance
TOOL_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}

async def safe_execute_tool(tool_function, tool_input: dict, tool_call_id: str) -> ToolMessage:
    """
    Executes an agent tool securely. Acts as a Pydantic Firewall to intercept 
    execution and schema errors, formatting them cleanly back into the agentic loop.
    """
    tool_name = tool_function.name
    
    # 1. Input Firewall: Catch corrupted JSON strings if the LLM bypassed dict formatting
    if isinstance(tool_input, str):
        try:
            tool_input = json.loads(tool_input)
        except json.JSONDecodeError as e:
            print(f"🛑 [FIREWALL INTERCEPT] JSONDecodeError on '{tool_name}'")
            return ToolMessage(
                content=f"Validation Error for tool '{tool_name}': Arguments were not valid JSON. Details: {str(e)}. Please correct your JSON formatting.",
                tool_call_id=tool_call_id,
                status="error"
            )

    try:
        # 2. Execution Layer
        print(f"🔥 [FIREWALL PASSED] Executing '{tool_name}' with args: {tool_input}")
        result = await tool_function.ainvoke(tool_input)
        
        return ToolMessage(
            content=str(result), 
            tool_call_id=tool_call_id,
            status="success"
        )

    # 3. Output/Validation Firewall: Catch specific Pydantic schema violations
    except ValidationError as e:
        error_details = e.errors()
        readable_errors = []
        for err in error_details:
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Invalid value")
            readable_errors.append(f"[{loc}]: {msg}")
        
        error_msg = "; ".join(readable_errors)
        print(f"🛑 [FIREWALL INTERCEPT] Pydantic ValidationError on '{tool_name}': {error_msg}")
        
        return ToolMessage(
            content=f"Schema Validation Error for tool '{tool_name}': {error_msg}. Please fix your argument types and try again.",
            tool_call_id=tool_call_id,
            status="error"
        )

    # 4. Standard Python Type/Value bounds (e.g. missing positional arguments)
    except (TypeError, ValueError) as e:
        print(f"🛑 [FIREWALL INTERCEPT] Python Type/Value Error on '{tool_name}': {str(e)}")
        return ToolMessage(
            content=f"Argument Error for tool '{tool_name}': {str(e)}. Ensure you are matching the tool's required parameters.",
            tool_call_id=tool_call_id,
            status="error"
        )

    # 5. Fallback for catastrophic internal tool failures (e.g. API timeouts)
    except Exception as e:
        print(f"💥 [RUNTIME CRASH] Unexpected crash inside tool '{tool_name}': {str(e)}")
        return ToolMessage(
            content=f"ERROR: Tool '{tool_name}' failed to execute internally. Reason: {str(e)}. Please attempt a different approach or tool fallback.",
            tool_call_id=tool_call_id,
            status="error"
        )