# app/tools/change_behaviour.py

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

# Force the LLM to choose from exact allowed strings
class StateTransitionSchema(BaseModel):
    new_profile: Literal["STANDARD_CODING", "DEEP_RESEARCH", "CASUAL_PRODUCTIVITY", "CRITICAL_REFLECTIVE"] = Field(
        description="The exact name of the profile you must transition to."
    )

@tool(args_schema=StateTransitionSchema)
async def change_behaviour_profile(new_profile: str) -> str:
    """
    CRITICAL: Call this tool IMMEDIATELY when the user's request requires a transition 
    to a different behavior profile (e.g., DEEP_RESEARCH or STANDARD_CODING).
    """
    # This string just acts as a confirmation bridge
    return f"SUCCESS: System transition authorized. Changing state to {new_profile}."