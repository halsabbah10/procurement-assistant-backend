"""Conversation sidebar metadata — title + timestamps only. The actual
message content lives in two places by design: the LangGraph checkpoint
(authoritative, used to reconstruct history in a fresh browser/session —
see app/agent/graph.py::reconstruct_conversation_messages) and the
frontend's localStorage cache (full-fidelity, including the query/chart/
suggestions enrichments that are never persisted server-side since they're
regenerated fresh per turn, not conversational state)."""

from __future__ import annotations

from datetime import UTC, datetime

from app.db.client import get_checkpoint_database


async def touch_conversation(thread_id: str, user_message: str) -> None:
    db = get_checkpoint_database()
    now = datetime.now(UTC)
    title = user_message.strip().replace("\n", " ")[:60]
    await db.conversations.update_one(
        {"_id": thread_id},
        {
            "$setOnInsert": {
                "title": title or "New conversation",
                "created_at": now,
                "is_deleted": False,
            },
            "$set": {"updated_at": now},
        },
        upsert=True,
    )
