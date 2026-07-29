from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import reconstruct_conversation_messages
from app.core.client_id import get_client_id
from app.db.client import get_checkpoint_database

router = APIRouter()


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


@router.get("/api/conversations")
async def list_conversations(limit: int = 50, client_id: str | None = Depends(get_client_id)):
    # No client_id (missing/malformed header) -> no conversations are
    # scoped to it, so the only correct answer is an empty list, not every
    # conversation from every visitor (the bug this replaces).
    if not client_id:
        return []
    db = get_checkpoint_database()
    cursor = (
        db.conversations.find({"is_deleted": {"$ne": True}, "client_id": client_id})
        .sort("updated_at", -1)
        .limit(min(limit, 100))
    )
    docs = await cursor.to_list(length=min(limit, 100))
    return [
        {
            "id": doc["_id"],
            "title": doc.get("title") or "New conversation",
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
        }
        for doc in docs
    ]


@router.patch("/api/conversations/{conversation_id}")
async def rename_conversation(
    conversation_id: str, body: RenameRequest, client_id: str | None = Depends(get_client_id)
):
    # A missing/malformed client_id must never match anything, including a
    # legacy conversation with no client_id field at all — Mongo's equality
    # match treats {"client_id": None} as "field is null OR absent", which
    # would otherwise let a header-less request touch every unowned
    # conversation. Short-circuit before querying.
    if not client_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db = get_checkpoint_database()
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    # Always 404 (never a distinct 403) on any ownership mismatch, matching
    # the existing "not found" response for a nonexistent id — doesn't leak
    # whether a given id belongs to someone else.
    result = await db.conversations.update_one(
        {"_id": conversation_id, "client_id": client_id}, {"$set": {"title": title}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"id": conversation_id, "title": title}


@router.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, client_id: str | None = Depends(get_client_id)):
    if not client_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db = get_checkpoint_database()
    result = await db.conversations.update_one(
        {"_id": conversation_id, "client_id": client_id}, {"$set": {"is_deleted": True}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found.")


@router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str, client_id: str | None = Depends(get_client_id)
):
    if not client_id:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    db = get_checkpoint_database()
    owned = await db.conversations.find_one(
        {"_id": conversation_id, "client_id": client_id}, {"_id": 1}
    )
    if owned is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    messages = await reconstruct_conversation_messages(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": messages}
