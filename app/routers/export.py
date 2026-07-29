import csv
import io
import json

from bson import ObjectId, json_util
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from app.core.rate_limit import RateLimiter
from app.db.client import get_database
from app.db.safe_pipeline import MAX_QUERY_TIME_MS, InvalidQueryError, parse_and_validate_pipeline
from app.schemas.export import ExportRequest

router = APIRouter()
# Separate, lighter budget than chat's limiter — this endpoint never calls
# an LLM, but still runs a MongoDB aggregation on shared free-tier infra and
# must not be hammered.
limiter = RateLimiter(per_minute=10, daily_cap=100)

# This app only ever stores purchase order data and its category embeddings
# in the `procurement` database — deliberately not derived from whatever
# collections happen to exist live, so a future new collection (audit logs,
# anything else) doesn't become silently exportable without a code change.
EXPORTABLE_COLLECTIONS = {"purchase_orders"}


def _sanitize(value):
    """Recursively convert ObjectId to str so both the CSV and JSON export
    paths deal with plain JSON-safe types instead of relying on each
    serializer's own (differing) handling of BSON types."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _rows_to_csv(rows: list[dict]) -> str:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flat = {
            k: (json.dumps(v, default=str) if isinstance(v, (dict, list)) else v)
            for k, v in row.items()
        }
        writer.writerow(flat)
    return buffer.getvalue()


@router.post("/api/export")
async def export_query(request: Request, body: ExportRequest):
    client_ip = request.client.host if request.client else "unknown"
    if not limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Export limit reached — try again shortly.")
    limiter.record(client_ip)

    try:
        collection, pipeline = parse_and_validate_pipeline(body.query, EXPORTABLE_COLLECTIONS)
    except InvalidQueryError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    db = get_database()
    raw_rows = (
        await db[collection].aggregate(pipeline, maxTimeMS=MAX_QUERY_TIME_MS).to_list(length=None)
    )
    rows = [_sanitize(row) for row in raw_rows]

    if not rows:
        raise HTTPException(status_code=404, detail="Query returned no results to export.")

    if body.format == "json":
        content = json_util.dumps(rows, indent=2)
        media_type = "application/json"
        filename = "export.json"
    else:
        content = _rows_to_csv(rows)
        media_type = "text/csv"
        filename = "export.csv"

    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
