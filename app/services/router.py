import os
from typing import List, Any
from dotenv import load_dotenv
from pydantic import SecretStr
from langchain_groq import ChatGroq

# Import your schemas and tools
from app.core.schema import ToolSelection
from app.tools.state_immutability import TOOL_REGISTRY

load_dotenv()
api_key = os.environ.get("GROQ_API_KEY")

async def semantic_tool_router(user_text: str, current_state: str) -> List[Any]:
    """
    Scans the global TOOL_REGISTRY and dynamically selects the best tools for the current prompt.
    """
    # 1. Build a list of all available tools and their descriptions for the LLM to read
    registry_info = "\n".join([f"- Name: {name} | Description: {tool.description}" for name, tool in TOOL_REGISTRY.items()])

    # 2. The Strict Routing Prompt
    routing_prompt = f"""
    CRITICAL SYSTEM INSTRUCTION: You are a strict JSON routing controller.
    The user is currently in state: {current_state}.
    User message: "{user_text}"
    
    Global Tool Registry:
    {registry_info}
    
   INSTRUCTIONS:
       1. ALWAYS include 'change_behaviour_profile' in your list.
       2. Select up to 2 additional tools that are highly relevant to the User message.
       3. IF YOU ARE UNSURE, OR IF NO TOOLS MATCH: Return ONLY ["change_behaviour_profile"]. Do not hallucinate tool names that do not exist in the Global Tool Registry.
    
    FINAL WARNING: You MUST output pure JSON. Do not add any extra keys.
    Your output MUST exactly match this JSON structure:
    {{"selected_tools": ["change_behaviour_profile", "other_tool"]}}
    
    """

    try:
        # 3. Use the fast model in JSON mode to act as the router
        router_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0, api_key=SecretStr(api_key))
        router = router_llm.with_structured_output(ToolSelection, method="json_mode")
        
        selection = await router.ainvoke(routing_prompt)
        selected_tool_names = selection.selected_tools
        print(f"🧠 [ROUTER] Selected tools: {selected_tool_names}")
        
        # 4. Map the string names back to actual LangChain Tool objects
        active_tools = []
        for name in selected_tool_names:
            if name in TOOL_REGISTRY:
                active_tools.append(TOOL_REGISTRY[name])
                
        # Failsafe: Ensure transition tool is always present
        if TOOL_REGISTRY["change_behaviour_profile"] not in active_tools:
            active_tools.append(TOOL_REGISTRY["change_behaviour_profile"])
            
        return active_tools

    except Exception as e:
        print(f"⚠️ Router failed, falling back to safe defaults: {e}")
        return [TOOL_REGISTRY["change_behaviour_profile"]]