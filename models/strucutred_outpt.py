# STRUCTURED OUTPUT — agent always returns this shape
from pydantic import BaseModel, Field
from typing import Optional


class AgentResponse(BaseModel):
    answer: str = Field(description="The main response to the user")
    files_modified: list[str] = Field(default=[], description="Files changed this turn")
    commands_run: list[str] = Field(default=[], description="Shell commands executed")
    follow_up_suggested: Optional[str] = Field(None, description="What to do next")
    confidence: float = Field(ge=0, le=1, description="How confident in the answer")


# Force the LLM to return this shape
from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(pydantic_object=AgentResponse)
