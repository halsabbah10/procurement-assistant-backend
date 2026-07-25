from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import reconstruct_conversation_messages
from app.db.client import get_checkpoint_database

router = APIRouter()


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=100)


@router.get("/api/conversations")
async def list_conversations(limit: int = 50):
    db = get_checkpoint_database()
    cursor = (
        db.conversations.find({"is_deleted": {"$ne": True}})
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
async def rename_conversation(conversation_id: str, body: RenameRequest):
    db = get_checkpoint_database()
    title = body.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty.")
    result = await db.conversations.update_one(
        {"_id": conversation_id}, {"$set": {"title": title}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"id": conversation_id, "title": title}


@router.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str):
    db = get_checkpoint_database()
    result = await db.conversations.update_one(
        {"_id": conversation_id}, {"$set": {"is_deleted": True}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Conversation not found.")


@router.get("/api/conversations/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    messages = await reconstruct_conversation_messages(conversation_id)
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return {"messages": messages}
