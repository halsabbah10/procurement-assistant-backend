"""Generate item embeddings and load them into item_embeddings with a
$vectorSearch index. Run after run_ingestion.py.

Usage: python scripts/run_embeddings.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch  # noqa: E402
from langchain_voyageai import VoyageAIEmbeddings  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.db.client import get_database  # noqa: E402
from app.db.embeddings import dedupe_items  # noqa: E402

INDEX_NAME = "item_vector_index"


async def generate_embeddings() -> None:
    settings = get_settings()
    db = get_database()

    rows = await db.purchase_orders.find(
        {},
        {
            "segment_title": 1,
            "family_title": 1,
            "class_title": 1,
            "commodity_title": 1,
        },
    ).to_list(length=None)
    items = dedupe_items(rows)
    print(f"Deduped to {len(items)} distinct UNSPSC categories from {len(rows)} rows.")

    await db.item_embeddings.delete_many({})

    embeddings = VoyageAIEmbeddings(model="voyage-4", voyage_api_key=settings.voyage_api_key)
    vector_store = MongoDBAtlasVectorSearch.from_connection_string(
        connection_string=settings.mongodb_uri,
        namespace=f"{settings.mongodb_db_name}.item_embeddings",
        embedding=embeddings,
        index_name=INDEX_NAME,
    )

    texts = [item["text"] for item in items]
    metadatas = [
        {
            "text": item["text"],
            "commodity_title": item["commodity_title"],
            "class_title": item["class_title"],
            "family_title": item["family_title"],
            "segment_title": item["segment_title"],
        }
        for item in items
    ]
    vector_store.add_texts(texts=texts, metadatas=metadatas)

    print(f"Loaded {len(items)} category embeddings into item_embeddings.")
    print(
        "NOTE: on Atlas (cloud or mongodb-atlas-local), create the vector search "
        f"index named '{INDEX_NAME}' on item_embeddings.embedding if "
        "add_texts did not create it automatically for this MongoDB version."
    )


if __name__ == "__main__":
    asyncio.run(generate_embeddings())
