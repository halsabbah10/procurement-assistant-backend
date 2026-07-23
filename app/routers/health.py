from fastapi import APIRouter

from app.db.client import get_database

router = APIRouter()


@router.get("/health/live")
async def liveness():
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness():
    db = get_database()
    await db.command("ping")
    return {"status": "ok"}
