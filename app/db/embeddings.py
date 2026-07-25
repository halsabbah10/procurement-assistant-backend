"""Dedup + embed distinct UNSPSC categories for the semantic search layer.

Embeds the distinct (segment_title, family_title, class_title, commodity_title)
hierarchy combinations, not raw item text. Two reasons, not just one:

1. Storage: ~13,295 distinct category combos vs. 246,859 distinct item-text
   combos — a ~19x reduction. The item-text approach needed ~2GB+ once
   embedded, blowing well past MongoDB Atlas's free M0 tier (512MB cap,
   confirmed by hitting the actual quota in production). The category
   approach fits comfortably with room to spare, at zero cost.
2. Quality: matching against clean taxonomy category names is less noisy
   than matching against raw item_name/item_description text, which is full
   of typos, abbreviations, and one-off product model numbers that don't
   generalize. "Network security software" (a commodity_title) is a more
   reliable semantic anchor than every differently-worded free-text variant
   of it across 346K rows.

The embedded text joins the full hierarchy (segment > family > class >
commodity) for context, since a bare commodity title can be ambiguous
without knowing which broader category it sits under.
"""
from __future__ import annotations


def dedupe_items(rows: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for row in rows:
        commodity_title = row.get("commodity_title")
        if not commodity_title:
            continue
        parts = [
            row.get("segment_title"),
            row.get("family_title"),
            row.get("class_title"),
            commodity_title,
        ]
        text = " > ".join(part for part in parts if part)
        if text not in seen:
            seen[text] = {
                "text": text,
                "commodity_title": commodity_title,
                "class_title": row.get("class_title"),
                "family_title": row.get("family_title"),
                "segment_title": row.get("segment_title"),
            }
    return list(seen.values())
