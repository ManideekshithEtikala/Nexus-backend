import os
from pydantic import SecretStr
from langchain_groq import ChatGroq

# Import your schemas
from app.core.schema import ScalableAgentResponseSchema, UIBlock

api_key = os.environ.get("GROQ_API_KEY")

# Temperature 0.0 forces strict adherence to our structural json definitions
extractor_llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.0, api_key=SecretStr(api_key)
)
structured_extractor = extractor_llm.with_structured_output(ScalableAgentResponseSchema)

async def generate_ui_pipeline(new_messages: list, fallback_text: str) -> ScalableAgentResponseSchema:
    """
    Takes the raw messages generated during the ReAct loop and forces them through 
    a strict JSON schema for the React frontend to render as UI Blocks.
    """
    recent_execution_history = "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in new_messages]
    )
    
    extraction_instruction = (
        f"CRITICAL SYSTEM INSTRUCTION: You are a strict JSON data parser, NOT a chatbot. "
        f"DO NOT include conversational filler, greetings, or explanations. "
        f"DO NOT output phrases like 'Here are the links' or 'Here is your table'.\n\n"
        
        f"Analyze the following operational execution logs. Break down the textual narrative thoughts, "
        f"tool observation reports, and metrics into an ordered sequence of visual UI component block elements matching these strict layout rules:\n\n"
        
        f"1. For 'markdown_text' blocks, supply your written response ONLY in the 'markdown_text' field.\n"
        f"2. For 'data_table' blocks, you must design a complete grid. Define the columns with unique ID values, "
        f"and map the rows as objects where the keys EXACTLY MATCH your column IDs. "
        f"Example: If column IDs are ['Category', 'Pros', 'Cons'], then your rows MUST look like: "
        f"[{{'Category': 'Definition', 'Pros': 'Scalability', 'Cons': 'Complexity'}}]. \n\n"
        
        f"Logs to process:\n{recent_execution_history}\n\n"
        f"FINAL WARNING: Output ONLY pure, valid JSON matching the schema. Any other text will crash the system."
    )
    
    try:
        final_structured_payload = await structured_extractor.ainvoke(extraction_instruction)
        return final_structured_payload
    except Exception as e:
        print(f"⚠️ Extraction Layer Fallback Triggered: {str(e)}")
        safe_text = str(fallback_text) if fallback_text else "Processing complete."
        return ScalableAgentResponseSchema(
            ui_pipeline=[UIBlock(block_type="markdown_text", markdown_text=safe_text)]
        )