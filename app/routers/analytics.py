from fastapi import APIRouter

from app.db.client import get_database

router = APIRouter()


@router.get("/api/analytics/summary")
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
            {"$group": {"_id": "$department_name", "total_spend": {"$sum": "$total_price"}}},
            {"$sort": {"total_spend": -1}},
            {"$limit": 10},
        ]
    ).to_list(length=None)

    return {
        "by_fiscal_year": [
            {"fiscal_year": r["_id"], "total_spend": r["total_spend"], "order_count": r["order_count"]}
            for r in by_fiscal_year
        ],
        "top_departments": [
            {"department": r["_id"], "total_spend": r["total_spend"]} for r in top_departments
        ],
    }
