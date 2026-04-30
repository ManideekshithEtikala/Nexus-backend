from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
import os


class ConversationSummarizer:
    """Generates and updates rolling bullet-point summaries."""
    
    def __init__(self):
        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("API_KEY"),
            temperature=0
        )
    
    async def summarize_batch(self, messages: list[BaseMessage]) -> str:
        """Create concise bullet-point summary from a batch of messages."""
        conv_text = self._format_messages(messages)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Create a concise bullet-point summary of this conversation segment.
Each bullet must be ONE line only, starting with '•'.
Focus ONLY on:
• Decisions made or conclusions reached
• Code created, modified, or deleted
• Errors encountered and solutions found
• User preferences or project requirements revealed

IGNORE greetings, thanks, casual chat, and repetitions.
OUTPUT ONLY bullet points — no explanations, no headings, no meta-commentary, no tags."""),
            ("human", "Conversation:\n{conversation}\n\nBULLET-POINT SUMMARY:")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({"conversation": conv_text})
        return result.content.strip()
    
    async def update_summary(
        self, 
        old_summary: str, 
        new_messages: list[BaseMessage]
    ) -> str:
        """Merge new messages into existing summary."""
        new_text = self._format_messages(new_messages)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """Update the existing summary by incorporating NEW information.
RULES:
1. KEEP existing bullets still relevant
2. MODIFY bullets with new developments  
3. ADD new bullets for new topics (max 1-2 lines each)
4. REMOVE bullets about resolved/irrelevant topics
5. Keep total bullets ≤ 10
6. Be extremely concise — one line per bullet
7. Output ONLY bullet points — no explanations, no metadata, no tags."""),
            ("human", "OLD SUMMARY:\n{old_summary}\n\nNEW MESSAGES:\n{new_messages}\n\nUPDATED SUMMARY:")
        ])
        
        chain = prompt | self.llm
        result = await chain.ainvoke({
            "old_summary": old_summary,
            "new_messages": new_text
        })
        return result.content.strip()
    
    def _format_messages(self, messages: list[BaseMessage]) -> str:
        """Format messages for LLM consumption."""
        lines = []
        for msg in messages:
            content = msg.content[:2000] if len(msg.content) > 2000 else msg.content
            prefix = "HUMAN" if msg.type == "human" else "AI"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)
