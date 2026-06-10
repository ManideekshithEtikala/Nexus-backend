import os
import json
import re
from pydantic import SecretStr, ValidationError
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Import your schemas
from app.core.schema import ScalableAgentResponseSchema, UIBlock

groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set")

api_key = SecretStr(groq_api_key)

# Plain LLM — NO .with_structured_output(), NO tools bound
extractor_llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key
)

_SYSTEM_PROMPT = """
You convert agent execution results into a clean UI JSON response.

Return exactly one valid JSON object and nothing else.

Valid schema:
{
  "ui_pipeline": [
    {
      "block_type": "markdown_text",
      "markdown_text": "string"
    }
  ]
}

Allowed blocks:

{
  "block_type": "markdown_text",
  "markdown_text": "string"
}

{
  "block_type": "data_table",
  "table_data": {
    "columns": [{"id": "source", "name": "Source"}],
    "rows": [{"source": "Example"}]
  }
}

{
  "block_type": "code_block",
  "language": "python",
  "code": "print(\\"hello\\")"
}

Rules:
- Output valid JSON only.
- No markdown fences.
- No comments.
- No extra text before or after the JSON.
- Escape all quotes inside string values.
- Combine all normal explanatory text into as few markdown_text blocks as possible.
- Do NOT split one answer into many tiny markdown blocks.
- Do NOT expose chain-of-thought, internal reasoning, or raw agent scratchpad text.
- Create a data_table only when the content is naturally tabular.
- Prefer this structure:
  1. one markdown_text block for the final answer
  2. optional data_table block
  3. optional code_block
"""
#helper function to strip unnecessary tags generated my llm
def _strip_artifacts(raw: str) -> str:
    """
    Cleans common LLM output artifacts before JSON parsing:
    - <function=Name>{...}</function> wrappers (the root cause of the 400 error)
    - ```json ... ``` markdown fences
    """
    # Remove <function=SchemaName>{...}</function> wrapper
    fn_match = re.search(r"<function=\w+>(.*?)</function>", raw, re.DOTALL)
    if fn_match:
        raw = fn_match.group(1)

    # Remove ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence_match:
        raw = fence_match.group(1)

    return raw.strip()

async def extract_json_from_text(raw: str) -> str:
    try:
        response = await extractor_llm.ainvoke([
            SystemMessage(content="Extract and return only the valid JSON object from the following text. Return JSON only."),
            HumanMessage(content=raw)
        ])
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        print(f"Error in extract_json_from_text: {e}")
        return raw
        
async def generate_ui_pipeline(new_messages: list, fallback_text: str) -> ScalableAgentResponseSchema:
    """
    Takes the raw messages generated during the ReAct loop and forces them through
    a strict JSON schema for the React frontend to render as UI Blocks.

    3-layer defence:
      Layer 1 — Ask the plain LLM to emit raw JSON (no function-calling layer).
      Layer 2 — Strip artifacts, parse JSON, validate with Pydantic manually.
      Layer 3 — If anything fails, return the agent's plain text as a markdown block.
    """
    recent_execution_history = "\n".join(
        [f"{msg.type.upper()}: {msg.content}" for msg in new_messages]
    )

    extraction_instruction = f"""
Transform the following agent execution content into a clean user-facing UI response.

Content:
{recent_execution_history}

Important:
- Produce a polished final answer for the user.
- Merge related text into one markdown_text block whenever possible.
- Do not include internal reasoning, tool traces, or repeated intermediate steps unless they are useful to the user.
- Only include a data_table if there is real structured data worth showing.
- Return valid JSON only.
""".strip()
    # ── Layer 1: Call the LLM without function-calling ────────────────────────
    try:
        response = await extractor_llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=extraction_instruction),
        ])

        raw_text = response.content if hasattr(response, "content") else str(response)
        repaired_text = await extract_json_from_text(raw_text)
        cleaned = _strip_artifacts(repaired_text)
        parsed_dict = json.loads(cleaned)
        validated = ScalableAgentResponseSchema(**parsed_dict)

        print("✅ [EXTRACTOR] Pipeline parsed and validated successfully.")
        return validated

    except (json.JSONDecodeError, ValidationError, TypeError, KeyError) as parse_err:
        print(f"⚠️ [EXTRACTOR] Parse/validation failed: {parse_err}")

    except Exception as llm_err:
        print(f"⚠️ [EXTRACTOR] LLM call failed: {llm_err}")

    print("🛟 [EXTRACTOR] Fallback triggered — returning plain markdown block.")
    safe_text = str(fallback_text) if fallback_text else "Processing complete."
    return ScalableAgentResponseSchema(
        ui_pipeline=[UIBlock(block_type="markdown_text", markdown_text=safe_text)]
    )