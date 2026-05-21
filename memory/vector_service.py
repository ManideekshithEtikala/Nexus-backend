import os
import uuid
import asyncio
from datetime import datetime
from typing import Optional, List
from pinecone import Pinecone
from google import genai
from google.genai import types
from core.config import settings


class VectorMemoryService:
    """
    Handles long-term semantic memory operations using Pinecone.
    Generates text embeddings in the cloud using Google's free AI Studio Tier.
    """

    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY
        self.index_name = settings.PINECONE_INDEX_NAME
        self.provider = settings.EMBEDDING_PROVIDER
        
        # Load embedding model & default to 'models/gemini-embedding-2' which is verified as active in this API environment
        self.model_name = settings.EMBEDDING_MODEL
            
        self.google_api_key = settings.GEMINI_API_KEY

        self.pc = None
        self.index = None
        self.google_client = None

        if not self.api_key or not self.google_api_key:
            print("\n" + "=" * 80)
            print(
                "[VectorMemoryService] WARNING: Missing PINECONE_API_KEY or GEMINI_API_KEY."
            )
            print("[VectorMemoryService] Vector Memory is running in OFFLINE mode.")
            print("=" * 80 + "\n")
            return

        try:
            # Connect to Pinecone
            print(
                f"[VectorMemoryService] Connecting to Pinecone..."
            )
            self.pc = Pinecone(api_key=self.api_key)
            
            # Check existing indexes and bootstrap if missing
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            if self.index_name not in existing_indexes:
                print(f"[VectorMemoryService] Index '{self.index_name}' was not found. Automatically creating a serverless index...")
                
                # Auto-determine dimension based on model configuration
                dimension = 768  # default for google text-embedding-004 and nomic-embed-text
                if self.provider == "openai":
                    dimension = 1536
                elif self.provider == "huggingface":
                    dimension = 384
                    
                from pinecone import ServerlessSpec
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                print(f"[VectorMemoryService] Serverless index '{self.index_name}' created successfully (Dimension: {dimension}, Metric: cosine).")

            self.index = self.pc.Index(self.index_name)

            # Connect to Google Cloud Embedding Service
            self.google_client = genai.Client(api_key=self.google_api_key)
            print(f"[VectorMemoryService] Cloud architecture connected successfully.")
        except Exception as e:
            print(f"[VectorMemoryService] Connection Error: {e}")
            self.index = None

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Generate vector embedding via Google Cloud API in a non-blocking thread."""
        if not text or not text.strip() or not self.google_client:
            return None

        try:
            # Run blocking API client call in a separate threadpool to keep FastAPI event loop free
            response = await asyncio.to_thread(
                self.google_client.models.embed_content,
                model=self.model_name,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                    if "Fact" in text
                    else "RETRIEVAL_QUERY",
                    output_dimensionality=768,  # Forces 768 dimensions to keep index lean
                )
            )
            # Pull values vector out of response
            return response.embeddings[0].values
        except Exception as e:
            print(f"[VectorMemoryService] Cloud embedding generation failed: {e}")
            return None

    async def upsert_session_summary(
        self, session_id: str, summary_text: str, user_id: str = "default_user"
    ):
        """Upserts a session summary to Pinecone cloud index."""
        if not self.index:
            return

        try:
            vector_text = f"Session Summary (Session {session_id}): {summary_text}"
            embedding = await self.get_embedding(vector_text)

            if not embedding:
                return

            doc_id = f"summary_{session_id}"
            metadata = {
                "type": "session_summary",
                "session_id": session_id,
                "user_id": user_id,
                "text": summary_text,
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.index.upsert(
                vectors=[{"id": doc_id, "values": embedding, "metadata": metadata}]
            )
            print(
                f"[VectorMemoryService] Saved session summary {session_id} to cloud storage."
            )
        except Exception as e:
            print(f"[VectorMemoryService] Error upserting session summary: {e}")

    async def upsert_fact(
        self,
        fact_content: str,
        source_session_id: str,
        user_id: str = "default_user",
        category: str = "general",
    ):
        """Appends an extracted long-term fact into Pinecone cloud index."""
        if not self.index:
            return

        try:
            vector_text = f"User preference/Fact: {fact_content}"
            embedding = await self.get_embedding(vector_text)

            if not embedding:
                return

            doc_id = f"fact_{uuid.uuid4().hex}"
            metadata = {
                "type": "key_fact",
                "session_id": source_session_id,
                "user_id": user_id,
                "text": fact_content,
                "category": category,
                "timestamp": datetime.utcnow().isoformat(),
            }

            self.index.upsert(
                vectors=[{"id": doc_id, "values": embedding, "metadata": metadata}]
            )
            print(
                f"[VectorMemoryService] Fact synchronized to Cloud VDB: '{fact_content[:40]}...'"
            )
        except Exception as e:
            print(f"[VectorMemoryService] Error upserting fact: {e}")

    async def query_relevant_context(
        self, query: str, user_id: str = "default_user", limit: int = 4
    ) -> str:
        """Queries Pinecone via cloud vectors to gather historical context."""
        if not self.index:
            return ""

        try:
            embedding = await self.get_embedding(query)
            if not embedding:
                return ""

            response = self.index.query(
                vector=embedding,
                top_k=limit,
                include_metadata=True,
                filter={"user_id": {"$eq": user_id}},
            )

            matches = response.get("matches", [])
            if not matches:
                return ""

            facts = []
            summaries = []

            for match in matches:
                metadata = match.get("metadata", {})
                doc_type = metadata.get("type")
                text = metadata.get("text", "")

                if not text:
                    continue

                if match.get("score", 0.0) < 0.60:
                    continue

                if doc_type == "key_fact":
                    facts.append(text)
                elif doc_type == "session_summary":
                    session_id = metadata.get("session_id", "unknown")
                    summaries.append(f"In past session '{session_id}': {text}")

            if not facts and not summaries:
                return ""

            context_lines = ["\n=== LONG-TERM CONTEXT FROM PAST CONVERSATIONS ==="]
            if facts:
                context_lines.append("\nLearned User Preferences & Facts:")
                for fact in set(facts):
                    context_lines.append(f"• {fact}")
            if summaries:
                context_lines.append("\nRelevant Past Decisions & Discussions:")
                for summary in set(summaries):
                    context_lines.append(f"• {summary}")
            context_lines.append("=================================================\n")

            return "\n".join(context_lines)

        except Exception as e:
            print(f"[VectorMemoryService] Error reading cloud memory: {e}")
            return ""
