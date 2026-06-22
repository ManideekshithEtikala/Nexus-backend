import json
import traceback
from typing import Dict, Any, Tuple
from pydantic import ValidationError
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from app.tools.Coding_tool import execute_agent_code_wasm
from app.tools.Web_search import web_search_tool
from app.tools.Research_docs import research_docs_tool
from app.tools.change_behaviour_tool import change_behaviour_profile
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import json

# 1. DEFINE SCHEMA AND TOOL FIRST
class SendEmailSchema(BaseModel):
    to_address: str = Field(
        description="The exact email address of the recipient."
    )
    subject: str = Field(
        description="A concise and professional subject line for the email."
    )
    body: str = Field(
        description="The complete markdown-formatted text body of the email."
    )

@tool("send_email", args_schema=SendEmailSchema)
def send_email_tool(to_address: str, subject: str, body: str) -> str:
    """
    Sends an email to a specified recipient. 
    Use this tool ONLY when the user explicitly requests to send an email, contact someone, or share a report via email.
    """
    print(f"\n" + "="*50)
    print(f"🚀 [EXTERNAL ACTION SIMULATION] Executing send_email_tool")
    print(f"TO: {to_address}")
    print(f"SUBJECT: {subject}")
    print(f"BODY:\n{body}")
    print("="*50 + "\n")

    return json.dumps({
        "status": "success",
        "message": f"Email successfully sent to {to_address}"
    })


# 2. ADD IT TO THE MASTER ARRAY
ALL_TOOLS = [
    web_search_tool,
    research_docs_tool,
    change_behaviour_profile,
    send_email_tool,
    execute_agent_code_wasm,
    # 👈 CRITICAL: Added here!
]

# 3. CREATE THE REGISTRY
TOOL_REGISTRY = {tool.name: tool for tool in ALL_TOOLS}


def validate_tool_input(tool: BaseTool, raw_input: Any) -> Tuple[bool, Any, str]:
    """
    Pre-Execution Firewall: Validates incoming arguments against a tool's 
    Pydantic schema BEFORE invocation. Resolves single responsibility routing.
    """
    # 1. Normalize JSON string inputs if the LLM bypassed formatting
    if isinstance(raw_input, str):
        try:
            raw_input = json.loads(raw_input)
        except json.JSONDecodeError as e:
            return False, None, f"Arguments were not valid JSON. Details: {str(e)}."

    # 2. Extract the tool's underlying Pydantic model schema
    # LangChain tools expose their schema via args_schema
    schema_model = tool.args_schema

    if schema_model:
        try:
            # Proactively validate data structurally against the Pydantic model
            validated_data = schema_model(**raw_input if isinstance(raw_input, dict) else {})
            # Return Python dictionary from validated Pydantic model
            return True, validated_data.model_dump(), ""
        except ValidationError as e:
            # Parse error locations and error definitions cleanly
            readable_errors = [
                f"[{' -> '.join(str(x) for x in err.get('loc', []))}]: {err.get('msg', 'Invalid value')}"
                for err in e.errors()
            ]
            return False, None, f"Schema Validation Error: {'; '.join(readable_errors)}."
            
    return True, raw_input, ""

async def safe_execute_tool(tool_function: BaseTool, tool_input: Any, tool_call_id: str) -> ToolMessage:
    """
    Executes an agent tool securely. Relies on decoupled pre-validation middleware 
    to guarantee a clean execution run.
    """
    tool_name = tool_function.name

    # Step 1: Run the Pre-Execution Firewall Check
    is_valid, sanitized_input, error_feedback = validate_tool_input(tool_function, tool_input)
    
    if not is_valid:
        print(f"🛑 [FIREWALL INTERCEPT] Validation failed for tool '{tool_name}': {error_feedback}")
        return ToolMessage(
            content=f"Error initializing tool '{tool_name}': {error_feedback} Please correct your input parameters and try again.",
            tool_call_id=tool_call_id,
            status="error"
        )

    # Step 2: Clean Execution Layer (We know the data type is guaranteed correct)
    try:
        print(f"🔥 [FIREWALL PASSED] Executing '{tool_name}' with safe args: {sanitized_input}")
        result = await tool_function.ainvoke(sanitized_input)
        
        return ToolMessage(
            content=str(result), 
            tool_call_id=tool_call_id,
            status="success"
        )

    except Exception as e:
        # This catch block ONLY handles actual runtime exceptions inside the tool (e.g. network timeout)
        print(f"💥 [RUNTIME CRASH] Internal tool error in '{tool_name}': {str(e)}")
        return ToolMessage(
            content=f"Runtime Error: Tool '{tool_name}' failed internally. Reason: {str(e)}.",
            tool_call_id=tool_call_id,
            status="error"
        )
