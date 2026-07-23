import csv
import io

import pytest
from motor.motor_asyncio import AsyncIOMotorClient

from app.db.ingest import clean_document

FIXTURE_CSV = """Creation Date,Purchase Date,Fiscal Year,LPA Number,Purchase Order Number,Requisition Number,Acquisition Type,Sub-Acquisition Type,Acquisition Method,Sub-Acquisition Method,Department Name,Supplier Code,Supplier Name,Supplier Qualifications,Supplier Zip Code,CalCard,Item Name,Item Description,Quantity,Unit Price,Total Price,Classification Codes,Normalized UNSPSC,Commodity Title,Class,Class Title,Family,Family Title,Segment,Segment Title,Location
08/27/2013,,2013-2014,7-12-70-26,REQ0011118,REQ0011118,IT Goods,,WSCA/Coop,,Consumer Affairs Department of,1740272,Pitney Bowes,,,NO,USB,USB,1,$1.00,$1.00,,,,,,,,,,
01/29/2014,,2013-2014,,REQ0011932,REQ0011932,NON-IT Goods,,Informal Competitive,,Consumer Affairs Department of,1760085,Rodea Auto Tech,,,NO,Tire Disposal,Tire Disposal,2,$2.00,$4.00,76121504,76121504,,,,,,,
"""


@pytest.mark.asyncio
async def test_ingestion_inserts_cleaned_documents():
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    db = client["procurement_test"]
    await db.purchase_orders.delete_many({})

    reader = csv.DictReader(io.StringIO(FIXTURE_CSV))
    docs = [clean_document(row) for row in reader]
    result = await db.purchase_orders.insert_many(docs)

    assert len(result.inserted_ids) == 2
    stored = await db.purchase_orders.find_one({"purchase_order_number": "REQ0011118"})
    assert stored["total_price"] == 1.00
    assert stored["quarter"] == 1

    await db.purchase_orders.delete_many({})
    client.close()
