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


# Matches langgraph-checkpoint-mongodb's MongoDBSaver default db_name — see
# app/agent/graph.py. Conversation metadata (title, timestamps) lives here
# too, alongside the checkpoints it describes, rather than in `procurement`
# (which stays exclusively the purchase-order dataset, partly to keep it
# clear of the Atlas free-tier storage budget documented in project memory).
CHECKPOINT_DB_NAME = "checkpointing_db"


def get_checkpoint_database() -> AsyncIOMotorDatabase:
    return _get_client()[CHECKPOINT_DB_NAME]
