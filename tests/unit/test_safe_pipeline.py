from datetime import UTC, datetime

import pytest
from bson import ObjectId

from app.db.safe_pipeline import (
    EXPORT_ROW_LIMIT,
    InvalidQueryError,
    parse_agent_pipeline,
    parse_and_validate_pipeline,
)

ALLOWED = {"purchase_orders"}


# --- Was this ever reachable via eval()? Confirm it's rejected now. -------


def test_rejects_dunder_import_payload():
    # The exact class of payload that exploited langchain_mongodb's
    # eval(agg_str, {"ObjectId": ..., ...}) — that globals dict omits
    # __builtins__, so Python auto-injects the real one, making __import__
    # reachable. ast.literal_eval never resolves a Name/Call node at all.
    payload = 'db.purchase_orders.aggregate([{"$match": __import__("os").environ}])'
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(payload, ALLOWED)


def test_rejects_function_call_in_pipeline():
    payload = 'db.purchase_orders.aggregate([{"$match": {"x": open("/etc/passwd").read()}}])'
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(payload, ALLOWED)


def test_rejects_arbitrary_name_reference():
    payload = 'db.purchase_orders.aggregate([{"$match": {"x": some_undefined_name}}])'
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(payload, ALLOWED)


# --- Real query shapes this app's agent actually produces -----------------


def test_parses_bare_unquoted_id_key():
    # _id as the $group key is the dominant shape for this app's
    # aggregations (see chart_spec.py's own docstring) and is idiomatic,
    # unquoted MongoDB-shell syntax — plain json.loads previously rejected
    # this outright.
    q = 'db.purchase_orders.aggregate([{"$group": {_id: "$department_name", "total": {"$sum": "$total_price"}}}])'
    collection, pipeline = parse_agent_pipeline(q, ALLOWED)
    assert collection == "purchase_orders"
    assert pipeline == [{"$group": {"_id": "$department_name", "total": {"$sum": "$total_price"}}}]


def test_parses_isodate():
    q = (
        'db.purchase_orders.aggregate([{"$match": '
        '{"creation_date": {"$gte": ISODate("2013-07-01T00:00:00Z")}}}])'
    )
    _, pipeline = parse_agent_pipeline(q, ALLOWED)
    assert pipeline[0]["$match"]["creation_date"]["$gte"] == datetime(2013, 7, 1, tzinfo=UTC)


def test_parses_new_date():
    q = 'db.purchase_orders.aggregate([{"$match": {"creation_date": {"$lt": new Date("2014-01-01T00:00:00Z")}}}])'
    _, pipeline = parse_agent_pipeline(q, ALLOWED)
    assert pipeline[0]["$match"]["creation_date"]["$lt"] == datetime(2014, 1, 1, tzinfo=UTC)


def test_parses_object_id_single_and_double_quoted():
    for quote in ('"', "'"):
        q = f"db.purchase_orders.aggregate([{{{quote}$match{quote}: {{{quote}_id{quote}: ObjectId({quote}507f1f77bcf86cd799439011{quote})}}}}])"
        _, pipeline = parse_agent_pipeline(q, ALLOWED)
        assert pipeline[0]["$match"]["_id"] == ObjectId("507f1f77bcf86cd799439011")


def test_parses_python_style_true_false_none():
    q = 'db.purchase_orders.aggregate([{"$match": {"calcard": True, "lpa_number": None, "flag": False}}])'
    _, pipeline = parse_agent_pipeline(q, ALLOWED)
    assert pipeline[0]["$match"] == {"calcard": True, "lpa_number": None, "flag": False}


def test_bare_id_conversion_does_not_mangle_unrelated_text():
    # The bare-_id regex must only match a standalone `_id` token, not
    # corrupt a string value that happens to contain "_id" as a substring.
    q = 'db.purchase_orders.aggregate([{"$match": {"item_description": "valid_id check"}}])'
    _, pipeline = parse_agent_pipeline(q, ALLOWED)
    assert pipeline[0]["$match"]["item_description"] == "valid_id check"


# --- Denylist: top-level and nested ----------------------------------------


@pytest.mark.parametrize(
    "stage",
    [
        "$out",
        "$merge",
        "$function",
        "$accumulator",
        "$where",
        "$currentOp",
        "$collStats",
        "$indexStats",
        "$planCacheStats",
        "$lookup",
        "$graphLookup",
        "$unionWith",
        "$facet",
        "$bucket",
        "$bucketAuto",
    ],
)
def test_denies_top_level_stage(stage):
    q = f'db.purchase_orders.aggregate([{{"{stage}": {{}}}}])'
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(q, ALLOWED)


def test_denies_lookup_nested_inside_facet():
    q = (
        'db.purchase_orders.aggregate([{"$facet": {"x": '
        '[{"$lookup": {"from": "purchase_orders", "pipeline": [], "as": "y"}}]}}])'
    )
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(q, ALLOWED)


def test_denies_where_nested_inside_match():
    q = 'db.purchase_orders.aggregate([{"$match": {"$where": "sleep(10000)"}}])'
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(q, ALLOWED)


# --- Collection allowlist ----------------------------------------------------


def test_rejects_collection_outside_allowlist():
    q = 'db.some_other_collection.aggregate([{"$match": {}}])'
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(q, ALLOWED)


def test_allows_any_collection_in_the_passed_allowlist():
    q = 'db.item_embeddings.aggregate([{"$match": {}}])'
    collection, _ = parse_agent_pipeline(q, {"purchase_orders", "item_embeddings"})
    assert collection == "item_embeddings"


# --- Malformed input ----------------------------------------------------


@pytest.mark.parametrize(
    "bad_command",
    [
        "not a command at all",
        "db.purchase_orders.aggregate(",
        "db.purchase_orders.aggregate([this is not valid syntax !!])",
        "db.purchase_orders.find({})",  # not .aggregate(
        'db.purchase_orders.aggregate("not a list")',
        "db.purchase_orders.aggregate([])",  # empty pipeline
    ],
)
def test_rejects_malformed_commands(bad_command):
    with pytest.raises(InvalidQueryError):
        parse_agent_pipeline(bad_command, ALLOWED)


# --- Export-specific behavior: forced row limit -----------------------------


def test_export_appends_row_limit():
    _, pipeline = parse_and_validate_pipeline(
        'db.purchase_orders.aggregate([{"$match": {}}])', ALLOWED
    )
    assert pipeline[-1] == {"$limit": EXPORT_ROW_LIMIT}


def test_export_denies_facet_bypass_of_row_limit():
    # $facet reshapes many documents into one, with unbounded nested
    # arrays — a $limit appended after it bounds document COUNT (1), not
    # the size of what's inside. Must be denied outright, not just
    # size-limited after the fact.
    q = 'db.purchase_orders.aggregate([{"$facet": {"dump": [{"$project": {"_id": 0}}]}}])'
    with pytest.raises(InvalidQueryError):
        parse_and_validate_pipeline(q, ALLOWED)


def test_export_denies_uncorrelated_lookup():
    q = (
        'db.purchase_orders.aggregate([{"$lookup": '
        '{"from": "purchase_orders", "pipeline": [], "as": "x"}}])'
    )
    with pytest.raises(InvalidQueryError):
        parse_and_validate_pipeline(q, ALLOWED)
