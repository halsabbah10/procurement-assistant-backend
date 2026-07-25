from fastapi import APIRouter, HTTPException

from app.core.ttl_cache import async_ttl_cache
from app.db.client import get_database

router = APIRouter()


@router.get("/api/analytics/summary")
@async_ttl_cache(ttl_seconds=300)
async def summary():
    db = get_database()
    pipeline = [
        {
            "$group": {
                "_id": "$fiscal_year",
                "total_spend": {"$sum": "$total_price"},
                "order_count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    by_fiscal_year = await db.purchase_orders.aggregate(pipeline).to_list(length=None)

    top_departments = await db.purchase_orders.aggregate(
        [
            {
                "$group": {
                    "_id": "$department_name",
                    "total_spend": {"$sum": "$total_price"},
                    "order_count": {"$sum": 1},
                }
            },
            {"$sort": {"total_spend": -1}},
            {"$limit": 10},
        ]
    ).to_list(length=None)

    by_acquisition_type = await db.purchase_orders.aggregate(
        [
            {
                "$group": {
                    "_id": "$acquisition_type",
                    "total_spend": {"$sum": "$total_price"},
                }
            },
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"total_spend": -1}},
            {"$limit": 8},
        ]
    ).to_list(length=None)

    top_suppliers = await db.purchase_orders.aggregate(
        [
            {
                "$group": {
                    "_id": "$supplier_name",
                    "total_spend": {"$sum": "$total_price"},
                    "order_count": {"$sum": 1},
                }
            },
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"total_spend": -1}},
            {"$limit": 10},
        ]
    ).to_list(length=None)

    by_quarter = await db.purchase_orders.aggregate(
        [
            {
                "$group": {
                    "_id": {"fiscal_year": "$fiscal_year", "quarter": "$quarter"},
                    "total_spend": {"$sum": "$total_price"},
                }
            },
            {"$sort": {"_id.fiscal_year": 1, "_id.quarter": 1}},
        ]
    ).to_list(length=None)

    return {
        "by_fiscal_year": [
            {"fiscal_year": r["_id"], "total_spend": r["total_spend"], "order_count": r["order_count"]}
            for r in by_fiscal_year
        ],
        "top_departments": [
            {"department": r["_id"], "total_spend": r["total_spend"], "order_count": r["order_count"]}
            for r in top_departments
        ],
        "by_acquisition_type": [
            {"acquisition_type": r["_id"], "total_spend": r["total_spend"]}
            for r in by_acquisition_type
        ],
        "top_suppliers": [
            {"supplier": r["_id"], "total_spend": r["total_spend"], "order_count": r["order_count"]}
            for r in top_suppliers
        ],
        "by_quarter": [
            {
                "fiscal_year": r["_id"]["fiscal_year"],
                "quarter": r["_id"]["quarter"],
                "total_spend": r["total_spend"],
            }
            for r in by_quarter
        ],
    }


@router.get("/api/analytics/department/{department_name}")
async def department_detail(department_name: str):
    db = get_database()
    match_stage = {"$match": {"department_name": department_name}}

    totals = await db.purchase_orders.aggregate(
        [
            match_stage,
            {
                "$group": {
                    "_id": None,
                    "total_spend": {"$sum": "$total_price"},
                    "order_count": {"$sum": 1},
                }
            },
        ]
    ).to_list(length=1)
    if not totals:
        raise HTTPException(status_code=404, detail=f"No records for department '{department_name}'.")

    by_fiscal_year = await db.purchase_orders.aggregate(
        [
            match_stage,
            {"$group": {"_id": "$fiscal_year", "total_spend": {"$sum": "$total_price"}}},
            {"$sort": {"_id": 1}},
        ]
    ).to_list(length=None)

    top_suppliers = await db.purchase_orders.aggregate(
        [
            match_stage,
            {
                "$group": {
                    "_id": "$supplier_name",
                    "total_spend": {"$sum": "$total_price"},
                    "order_count": {"$sum": 1},
                }
            },
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"total_spend": -1}},
            {"$limit": 8},
        ]
    ).to_list(length=None)

    top_categories = await db.purchase_orders.aggregate(
        [
            match_stage,
            {
                "$group": {
                    "_id": "$commodity_title",
                    "total_spend": {"$sum": "$total_price"},
                }
            },
            {"$match": {"_id": {"$ne": None}}},
            {"$sort": {"total_spend": -1}},
            {"$limit": 8},
        ]
    ).to_list(length=None)

    return {
        "department": department_name,
        "total_spend": totals[0]["total_spend"],
        "order_count": totals[0]["order_count"],
        "by_fiscal_year": [
            {"fiscal_year": r["_id"], "total_spend": r["total_spend"]} for r in by_fiscal_year
        ],
        "top_suppliers": [
            {"supplier": r["_id"], "total_spend": r["total_spend"], "order_count": r["order_count"]}
            for r in top_suppliers
        ],
        "top_categories": [
            {"category": r["_id"], "total_spend": r["total_spend"]} for r in top_categories
        ],
    }
