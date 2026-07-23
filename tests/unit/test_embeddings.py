from app.db.embeddings import dedupe_items


def test_dedupe_items_collapses_identical_item_text():
    rows = [
        {"item_name": "USB", "item_description": "USB", "commodity_title": None},
        {"item_name": "USB", "item_description": "USB", "commodity_title": None},
        {"item_name": "Tire Disposal", "item_description": "Tire Disposal", "commodity_title": None},
    ]
    deduped = dedupe_items(rows)
    assert len(deduped) == 2
    texts = {d["text"] for d in deduped}
    assert "USB USB" in texts
    assert "Tire Disposal Tire Disposal" in texts


def test_dedupe_items_skips_rows_with_no_item_name():
    rows = [{"item_name": None, "item_description": "x", "commodity_title": None}]
    assert dedupe_items(rows) == []
