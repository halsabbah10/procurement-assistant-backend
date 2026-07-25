from app.db.embeddings import dedupe_items

# Real UNSPSC hierarchy values observed in the dataset, not synthetic examples.
JALAPENO_ROW = {
    "segment_title": "Food Beverage and Tobacco Products",
    "family_title": "Fresh vegetables",
    "class_title": "Peppers",
    "commodity_title": "Jalapeno peppers",
}
VPN_SOFTWARE_ROW = {
    "segment_title": "Information Technology Broadcasting and Telecommunications",
    "family_title": "Software",
    "class_title": "Security and protection software",
    "commodity_title": "Network security or virtual private network VPN management software",
}


def test_dedupe_items_collapses_identical_category_hierarchy():
    rows = [JALAPENO_ROW, dict(JALAPENO_ROW), VPN_SOFTWARE_ROW]
    deduped = dedupe_items(rows)
    assert len(deduped) == 2
    texts = {d["text"] for d in deduped}
    assert (
        "Food Beverage and Tobacco Products > Fresh vegetables > Peppers > Jalapeno peppers"
        in texts
    )
    assert (
        "Information Technology Broadcasting and Telecommunications > Software > "
        "Security and protection software > Network security or virtual private "
        "network VPN management software"
    ) in texts


def test_dedupe_items_preserves_hierarchy_fields_as_metadata():
    deduped = dedupe_items([JALAPENO_ROW])
    assert deduped[0]["commodity_title"] == "Jalapeno peppers"
    assert deduped[0]["class_title"] == "Peppers"
    assert deduped[0]["family_title"] == "Fresh vegetables"
    assert deduped[0]["segment_title"] == "Food Beverage and Tobacco Products"


def test_dedupe_items_skips_rows_with_no_commodity_title():
    rows = [
        {"segment_title": "x", "family_title": "y", "class_title": "z", "commodity_title": None}
    ]
    assert dedupe_items(rows) == []


def test_dedupe_items_handles_missing_upper_hierarchy_gracefully():
    # A row with a commodity_title but missing segment/family/class shouldn't
    # crash or produce a text starting with stray " > " separators.
    rows = [
        {
            "segment_title": None,
            "family_title": None,
            "class_title": None,
            "commodity_title": "Jalapeno peppers",
        }
    ]
    deduped = dedupe_items(rows)
    assert deduped[0]["text"] == "Jalapeno peppers"
