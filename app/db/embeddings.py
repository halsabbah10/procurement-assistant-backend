"""Dedup + embed distinct item text for the semantic search layer.

Deliberately embeds distinct (item_name, item_description, commodity_title)
combinations, not all 346K rows — many rows share identical item text, and
deduping keeps this comfortably inside Voyage AI's free 200M-token tier.
"""
from __future__ import annotations


def dedupe_items(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in rows:
        if not row.get("item_name"):
            continue
        text = " ".join(
            part for part in (row.get("item_name"), row.get("item_description")) if part
        )
        if text not in seen:
            seen[text] = {"text": text, "commodity_title": row.get("commodity_title")}
    return list(seen.values())
