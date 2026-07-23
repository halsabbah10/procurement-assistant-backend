"""Idempotent ETL: CSV -> purchase_orders collection.

Usage: python scripts/run_ingestion.py /path/to/PURCHASE_ORDER_DATA_EXTRACT.csv
Safe to re-run: drops and recreates the collection + indexes each time.
"""
import asyncio
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.client import get_database  # noqa: E402
from app.db.ingest import clean_document  # noqa: E402

BATCH_SIZE = 2000


async def ingest(csv_path: str) -> None:
    db = get_database()
    await db.purchase_orders.delete_many({})

    csv.field_size_limit(sys.maxsize)
    batch: list[dict] = []
    total = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch.append(clean_document(row))
            if len(batch) >= BATCH_SIZE:
                await db.purchase_orders.insert_many(batch)
                total += len(batch)
                print(f"Inserted {total} rows...")
                batch = []
        if batch:
            await db.purchase_orders.insert_many(batch)
            total += len(batch)

    print(f"Ingestion complete: {total} documents.")
    print("Creating indexes...")
    await db.purchase_orders.create_index("creation_date")
    await db.purchase_orders.create_index("fiscal_year")
    await db.purchase_orders.create_index("department_name")
    await db.purchase_orders.create_index("acquisition_type")
    await db.purchase_orders.create_index("supplier_name")
    await db.purchase_orders.create_index([("fiscal_year", 1), ("quarter", 1)])
    print("Indexes created.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/run_ingestion.py <path-to-csv>")
        sys.exit(1)
    asyncio.run(ingest(sys.argv[1]))
