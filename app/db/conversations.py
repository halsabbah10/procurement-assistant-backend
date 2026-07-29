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


async def touch_conversation(thread_id: str, user_message: str, client_id: str | None) -> None:
    """`client_id` is an anonymous, client-generated identifier (see
    frontend/src/lib/api.ts) — this app has no login system, so it's the
    only scoping available. Set once via $setOnInsert (the conversation's
    owner never changes on later turns) and left unset if the caller sent
    none, which is a safe degrade: an unowned conversation simply never
    matches any client_id's list filter (see list_conversations_for_client)
    rather than being visible to everyone, which was the actual bug this
    replaces — see project audit, conversation cross-visitor visibility."""
    db = get_checkpoint_database()
    now = datetime.now(UTC)
    title = user_message.strip().replace("\n", " ")[:60]
    set_on_insert = {
        "title": title or "New conversation",
        "created_at": now,
        "is_deleted": False,
    }
    if client_id:
        set_on_insert["client_id"] = client_id
    await db.conversations.update_one(
        {"_id": thread_id},
        {
            "$setOnInsert": set_on_insert,
            "$set": {"updated_at": now},
        },
        upsert=True,
    )
