from functools import lru_cache

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import get_settings


@lru_cache
def _get_client() -> AsyncIOMotorClient:
    settings = get_settings()
    return AsyncIOMotorClient(settings.mongodb_uri)


def get_database() -> AsyncIOMotorDatabase:
    settings = get_settings()
    return _get_client()[settings.mongodb_db_name]
