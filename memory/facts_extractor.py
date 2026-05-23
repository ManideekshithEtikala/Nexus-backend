import os
import json

# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class FactsExtractor:
    """
    Analyzes conversation turns to extract high-value personal user preferences,
    development environment settings, architectural decisions, and tech stack details.
    """

    def __init__(self):
        self.llm = ChatGroq(
            model="qwen/qwen3-32b",
            api_key=os.getenv("API_KEY"),
            temperature=0,
            max_tokens=1024  # Provide enough token budget for reasoning + JSON array
        )

    async def extract_new_facts(self, user_msg: str, assistant_ans: str) -> list[str]:
        """
        Analyzes the current conversation turn to see if the user revealed
        any notable facts, settings, or development details.
        Returns a list of extracted fact strings.
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a highly analytical memory extraction system. Your job is to extract ANY notable and persistent facts about the user's setup, preferences, frameworks, databases, project goals, OR personal identity from this single conversation turn.

Look specifically for:
1. User profile and personal identity details (e.g., their name, background, job role, company, what they are currently doing or did in the past).
2. Tech stack choices (e.g., "Using FastAPI with Uvicorn", "Frontend in Next.js")
3. Developer styling/coding preferences (e.g., "Hates Tailwind CSS", "Prefers absolute imports")
4. Local environment facts (e.g., "Postgres is running locally on port 5432")
5. Important structural decisions made (e.g., "Agreed to use Pinecone for long term memory")

CRITICAL RULES:
- Extract user identity facts like name, background, role, and company permanently (e.g., "The user's name is Mani Deekshith" or "The user works as an AI researcher at NVIDIA"). These are NOT fluff and must be retained forever.
- Do NOT extract conversational fluff, temporary greetings, transient moods, or minor conversational filler.
- Be extremely concise. Formulate each fact as a simple declarative sentence in third person (e.g., "The user's name is X", "The user is a software developer", "User prefers vanilla CSS over Tailwind" or "Local PostgreSQL database is named 'chat_bot'").
- Return the extracted facts strictly as a JSON array of strings. Example format: ["Fact 1", "Fact 2"]
- If no persistent facts or preferences are revealed, return an empty JSON array: []
- Output ONLY the raw JSON array — NO explanations, NO Markdown formatting, NO comments.
- CRITICAL: Do NOT output `<think>` or reasoning thoughts if possible, but if you do, ensure you always close it with `</think>` and print the clean JSON array at the very end.""",
                ),
                (
                    "human",
                    "User: {user_msg}\nAssistant: {assistant_ans}\n\nEXTRACTED FACTS:",
                ),
            ]
        )

        try:
            chain = prompt | self.llm

            # Pass variables inside the execution map to fill prompt variables properly
            result = await chain.ainvoke(
                {"user_msg": user_msg, "assistant_ans": assistant_ans}
            )
            content = result.content.strip()

            print(f"[FactsExtractor] Raw LLM response content: {content}")

            # Strip reasoning think tags if present (common in Qwen reasoning variants)
            if "<think>" in content and "</think>" in content:
                content = content.split("</think>")[1].strip()

            # Clean markdown code blocks if the model returned them
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()

            facts = json.loads(content)
            if isinstance(facts, list):
                # Clean facts list and remove empty entries
                cleaned_facts = [str(f).strip() for f in facts if str(f).strip()]
                print(
                    f"[FactsExtractor] Extracted {len(cleaned_facts)} new facts from turn."
                )
                return cleaned_facts
            return []
        except Exception as e:
            print(f"[FactsExtractor] Error extracting facts: {e}")
            return []
