import os
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from app.graph.supervisor_agent.state import BaseAgentState


async def compress_memory_node(state: BaseAgentState) -> dict:
    """
    Compresses historical messages into a rolling summary once history exceeds 6 messages,
    and updates the active context_window with only the latest 6 messages.
    """
    current_summary = state.get("summary", "")
    current_messages = state.get("messages", [])

    # 🎯 FIX 1: If history is short, context window is simply all current messages
    if len(current_messages) <= 6:
        return {"summary": current_summary, "context_window": current_messages}
    print(
        f"🗜️ [COMPRESS MEMORY] Compressing memory: current_summary={current_summary!r}"
        f"current_messages={len(current_messages)} messages"
    )
    # Slice the history cleanly
    msgs_to_summarize = current_messages[:-6]
    sliding_window_messages = current_messages[-6:]

    # 🎯 FIX 2: Create a clean prompt structure
    system_instruction = (
        "You are an expert memory compression engine.\n"
        "Your task is to take the existing summary of a conversation and update it "
        "by integrating the new segment of messages provided by the user.\n"
        "Keep the summary dense, direct, and factual. Avoid jargon or conversational fluff.\n\n"
        f"Current Summary:\n{current_summary}"
    )

    # Combine system instructions with the actual historical messages to process
    llm_payload = [SystemMessage(content=system_instruction)] + msgs_to_summarize

    # Use a low-latency model for compression tasks (e.g., llama3-8b-8192 on Groq)
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=os.environ.get("GROQ_API_KEY"),
        temperature=0.1,  # Lower temperature is better for purely factual summarization
    )

    try:
        result = await llm.ainvoke(llm_payload)
        updated_summary = result.content

        optimized_context = [
            SystemMessage(
                content=f"Summary of prior conversation logs:\n{updated_summary}"
            )
        ] + sliding_window_messages
        print(f"🗜️[CURRENTLY USING SUMMARY] Updated summary: {updated_summary[:100]}...")
        return {"summary": updated_summary, "context_window": optimized_context}

    except Exception as e:
        print(f"❌ Error invoking Groq model during compression: {e}")
        # Fallback gracefully: keep old summary, but still enforce the sliding window limit
        fallback_context = [
            SystemMessage(
                content=f"Summary of prior conversation logs:\n{current_summary}"
            )
        ] + sliding_window_messages

        return {"summary": current_summary, "context_window": fallback_context}
