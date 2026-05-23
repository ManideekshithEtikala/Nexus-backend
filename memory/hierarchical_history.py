from typing import Optional, List
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory

# pyrefly: ignore [missing-import]
from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
from .summarizer import ConversationSummarizer


# Configuration
SHORT_TERM_WINDOW = 6  # Keep last 6 messages raw in context
SUMMARY_BATCH_SIZE = 4  # Summarize in groups of 4 exchanges (8 messages)


class HierarchicalSQLChatMessageHistory(BaseChatMessageHistory):
    """
    Combines short-term raw messages with rolling bullet-point summaries.
    - Last 6 messages: kept verbatim
    - Older messages: represented as bullet-point summary
    - Summary updates incrementally as new messages arrive
    """

    def __init__(
        self,
        session_id: str,
        connection,
        table_name: str = "message_store",
        async_mode: bool = True,
    ):
        self.session_id = session_id
        self.sql_history = SQLChatMessageHistory(
            session_id=session_id,
            connection=connection,
            table_name=table_name,
            async_mode=async_mode,
        )
        self.summarizer = ConversationSummarizer()
        self._cached_summary: Optional[str] = None

    @property
    def messages(self) -> list[BaseMessage]:
        """Synchronous messages property."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            return []
        return loop.run_until_complete(self.aget_messages())

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Synchronous add_messages."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            loop.create_task(self.aadd_messages(messages))
        else:
            loop.run_until_complete(self.aadd_messages(messages))

    def clear(self) -> None:
        """Synchronous clear."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            loop.create_task(self.aclear())
        else:
            loop.run_until_complete(self.aclear())

    async def aclear(self) -> None:
        """Asynchronously clear the history."""
        await self.sql_history.aclear()
        self._cached_summary = None

    async def aget_messages(self) -> list[BaseMessage]:
        """Load messages with rolling summary injected at the front."""
        all_messages = await self.sql_history.aget_messages()

        print(
            f"[Hierarchical] Session {self.session_id}: Total messages = {len(all_messages)}"
        )

        if len(all_messages) <= SHORT_TERM_WINDOW:
            print(
                f"[Hierarchical] All messages fit in short-term window, no summary needed"
            )
            return all_messages

        recent = all_messages[-SHORT_TERM_WINDOW:]
        older = all_messages[:-SHORT_TERM_WINDOW]
        print(f"[Hierarchical] Split: recent={len(recent)}, older={len(older)}")

        summary = await self._get_or_create_summary(older)

        if summary:
            summary_msg = SystemMessage(
                content=f"Previous conversation highlights (before the last {SHORT_TERM_WINDOW} messages):\n{summary}",
                additional_kwargs={"type": "rolling_summary"},
            )
            result = [summary_msg] + recent
            print(
                f"[Hierarchical] Returning {len(result)} messages (1 summary + {len(recent)} raw)"
            )
            return result

        print("[Hierarchical] No summary generated, returning recent only")
        return recent

    async def aadd_messages(self, messages: list[BaseMessage]):
        """Add messages and update summary if needed."""
        await self.sql_history.aadd_messages(messages)
        self._cached_summary = None  # Invalidate — will regenerate on next read

    async def _get_or_create_summary(self, older_messages: list[BaseMessage]) -> str:
        """Get summary from cache/DB or create new one from older messages."""
        # Try cache first
        if self._cached_summary:
            return self._cached_summary

        # Try loading from DB (summary text + how many messages it covers)
        summary, covered_count = await self._load_summary_from_db()

        if summary and covered_count >= len(older_messages):
            # Summary already covers all older messages
            self._cached_summary = summary
            return summary

        # Need to create or extend summary
        if summary:
            # Existing summary covers only part — update incrementally
            uncovered = older_messages[covered_count:]
            summary = await self.summarizer.update_summary(summary, uncovered)
            # Save with new covered count
            await self._persist_summary(summary, len(older_messages))
        else:
            # First time — create fresh summary
            summary = await self.summarizer.summarize_batch(older_messages)
            await self._persist_summary(summary, len(older_messages))

        self._cached_summary = summary
        return summary

    async def _load_summary_from_db(self) -> tuple[Optional[str], int]:
        """Fetch latest summary and covered message count from DB."""
        try:
            from sqlalchemy.ext.asyncio import AsyncSession
            from sqlalchemy import text
            from database.database import async_engine

            async with AsyncSession(async_engine) as session:
                result = await session.execute(
                    text("""
                        SELECT summary_text, covered_message_count 
                        FROM conversation_summaries 
                        WHERE session_id = :sid 
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """),
                    {"sid": self.session_id},
                )
                row = result.first()
                if row:
                    return row[0], row[1]  # (summary_text, covered_count)
                return None, 0
        except Exception as e:
            import traceback

            print(f"[HierarchicalHistory] Error loading summary: {e}")
            traceback.print_exc()
            return None, 0

    async def _persist_summary(self, summary: str, covered_count: int):
        """Save summary to conversation_summaries table."""
        try:
            from sqlalchemy.ext.asyncio import AsyncSession
            from sqlalchemy import text
            from database.database import async_engine

            async with AsyncSession(async_engine) as session:
                await session.execute(
                    text("""
                        INSERT INTO conversation_summaries 
                        (session_id, summary_text, covered_message_count, updated_at)
                        VALUES (:sid, :summary, :count, CURRENT_TIMESTAMP)
                    """),
                    {
                        "sid": self.session_id,
                        "summary": summary,
                        "count": covered_count,
                    },
                )
                await session.commit()
        except Exception as e:
            print(f"Warning: Could not persist summary: {e}")


