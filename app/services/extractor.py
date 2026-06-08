import os
import json
import re
from pydantic import SecretStr, ValidationError
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Import your schemas
from app.core.schema import ScalableAgentResponseSchema, UIBlock

groq_api_key = os.environ.get("GROQ_API_KEY")
api_key = SecretStr(groq_api_key)

# ─────────────────────────────────────────────────────────────────────────────
# WHY WE NO LONGER USE .with_structured_output():
#
#   .with_structured_output() routes through Groq's function-calling layer.
#   Llama models occasionally emit:  <function=SchemaName>{json}</function>
#   instead of a proper tool-call response. Groq's API validator rejects that
#   with:  400 tool_use_failed — crashing the entire request.
#
#   This happens non-deterministically in production (longer context, richer
#   web search content) but not locally (short, simple prompts).
#
#   FIX: Prompt the model to return raw JSON as plain text, then parse +
#   validate it ourselves with Pydantic. No function-calling layer involved.
# ─────────────────────────────────────────────────────────────────────────────

# Plain LLM — NO .with_structured_output(), NO tools bound
extractor_llm = ChatGroq(
    model="llama-3.3-70b-versatile", temperature=0.0, api_key=api_key
)

_SYSTEM_PROMPT = """
You are a strict JSON data parser. Output ONLY a single raw JSON object — no markdown fences, no explanation, no preamble.

The JSON object must match this exact shape:
{
  "ui_pipeline": [
    // one or more block objects
  ]
}

Each block must have a "block_type" field. Allowed blocks:

1. markdown_text block:
   {"block_type": "markdown_text", "markdown_text": "<your markdown string>"}

2. data_table block:
   {
     "block_type": "data_table",
     "table_data": {
       "columns": [{"id": "Col1", "name": "Col1"}, {"id": "Col2", "name": "Col2"}],
       "rows": [{"Col1": "value", "Col2": "value"}]
     }
   }
   RULE: Row keys MUST exactly match column IDs.

3. code_block block:
   {"block_type": "code_block", "language": "python", "code": "<code string>"}

CRITICAL RULES:
- Output ONLY the raw JSON object. Nothing else.
- NEVER wrap output in <function=...> tags.
- NEVER add text before or after the JSON.
""".strip()

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
        f"FINAL WARNING: Output ONLY the raw JSON object. Any other text will crash the system."
    )

    # ── Layer 1: Call the LLM without function-calling ────────────────────────
    try:
        response = await extractor_llm.ainvoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=extraction_instruction),
        ])

        raw_text = response.content if hasattr(response, "content") else str(response)

        # ── Layer 2: Clean → parse → validate ────────────────────────────────
        try:
            cleaned = _strip_artifacts(raw_text)
            parsed_dict = json.loads(cleaned)
            validated = ScalableAgentResponseSchema(**parsed_dict)
            print("✅ [EXTRACTOR] Pipeline parsed and validated successfully.")
            return validated

        except (json.JSONDecodeError, ValidationError, TypeError, KeyError) as parse_err:
            print(f"⚠️ [EXTRACTOR] Parse/validation failed: {parse_err}")
            # Fall through to Layer 3

    except Exception as llm_err:
        print(f"⚠️ [EXTRACTOR] LLM call failed: {llm_err}")
        # Fall through to Layer 3

    # ── Layer 3: Safe fallback — never crash the endpoint ────────────────────
    print("🛟 [EXTRACTOR] Fallback triggered — returning plain markdown block.")
    safe_text = str(fallback_text) if fallback_text else "Processing complete."
    return ScalableAgentResponseSchema(
        ui_pipeline=[UIBlock(block_type="markdown_text", markdown_text=safe_text)]
    )