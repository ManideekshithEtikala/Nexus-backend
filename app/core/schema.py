from enum import Enum
from typing import List, Any, Optional, Literal, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage

class BehaviourPattern(str, Enum):
    STANDARD_CODING = "STANDARD_CODING"
    DEEP_RESEARCH = "DEEP_RESEARCH"
    CASUAL_PRODUCTIVITY = "CASUAL_PRODUCTIVITY"
    CRITICAL_REFLECTIVE = "CRITICAL_REFLECTIVE"

class AgentState(BaseModel):
    session_id: str
    messages: List[BaseMessage] = Field(default_factory=list)
    current_summary: Optional[str] = ""
    is_user_msg_important: bool = False
    is_ai_msg_important: bool = False
    new_agent_messages: List[BaseMessage] = Field(default_factory=list)
    current_behaviour: BehaviourPattern = Field(default=BehaviourPattern.CASUAL_PRODUCTIVITY)

    model_config = {"arbitrary_types_allowed": True}

class UserMessage(BaseModel):
    message: str
    sessionId: str

class ResumeAction(BaseModel):
    sessionId: str
    tool_name: str
    tool_args: dict
    tool_call_id: str
    is_approved: bool

class StandardizedRow(BaseModel):
    id: str = Field(description="The row item index or identifier")
    content: str = Field(description="The text content, description, value, or benefit")

class TableColumn(BaseModel):
    id: str = Field(description="The unique key used for this column")
    name: str = Field(description="The human-readable column display header title")

class ScalableTable(BaseModel):
    columns: List[TableColumn] = Field(description="The structural layout columns definition.")
    rows: List[Dict[str, Any]] = Field(description="List of row objects mapped to column IDs.")

class UIBlock(BaseModel):
    block_type: Literal["markdown_text", "citations", "data_table", "action_status"] = Field(description="The target component identifier widget mapping rules.")
    markdown_text: Optional[str] = Field(default=None, description="Use ONLY if block_type is 'markdown_text'")
    table_data: Optional[ScalableTable] = Field(default=None, description="Use ONLY if block_type is 'data_table'")

class ScalableAgentResponseSchema(BaseModel):
    ui_pipeline: List[UIBlock] = Field(description="An ordered serial list of UI presentation blocks.")

class ToolSelection(BaseModel):
    selected_tools: List[str] = Field(description="A list of the tool names selected from the registry. MUST always include 'change_behaviour_profile'.")

class MemoryTaggingSchema(BaseModel):
    is_Important: bool = Field(description="True context if text metadata should be permanently pinned.")
    
    #graph schema models
class GraphNode(BaseModel):
    entity_name: str = Field(description="The specific name of the entity (e.g., 'Alex', 'Max', 'Python').")
    
    # 🔓 OPEN ONTOLOGY: Instruct HOW to categorize, not WHAT to categorize
    entity_type: str = Field(
        description="Dynamically infer the generic Category of the entity in PascalCase (e.g., 'User', 'Dog', 'FamilyMember', 'Technology')."
    )

class GraphEdge(BaseModel):
    source_node: str = Field(description="The entity_name of the source node.")
    
    # 🔓 OPEN ONTOLOGY: Enforce syntax, allow semantic freedom
    relation: str = Field(
        description="Dynamically infer the relationship verb. MUST be generalized, in UPPERCASE_SNAKE_CASE, and maximum 3 words (e.g., 'BUILDS_WITH', 'LOVES', 'HATES', 'WORKS_AT')."
    )
    
    target_node: str = Field(description="The entity_name of the target node.")
class KnowledgeGraphUpdate(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]