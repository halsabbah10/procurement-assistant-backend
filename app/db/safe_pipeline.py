"""Safely parse and validate a `db.purchase_orders.aggregate([...])` command
string for re-execution outside the agent (the CSV/JSON export endpoint).

This deliberately does NOT reuse langchain_mongodb's
`MongoDBDatabase._parse_command`, which parses the pipeline with
`eval(agg_str, {"ObjectId": ..., "datetime": ..., "timezone": ...})`. That
call omits `__builtins__` from the globals dict, which does not sandbox
eval() — Python auto-injects the real `__builtins__` into any globals dict
that doesn't already have the key, so that code path has access to
`__import__`, `open`, etc. It's an acceptable-risk internal detail when the
input is the model's own tool-call argument, but it must never be reachable
from a client-supplied string, which is exactly what an export endpoint
that "re-runs the query shown in the UI" would do. This parser uses
`json.loads` only — structurally incapable of executing code — and
additionally denies any aggregation stage that can write data, run
server-side JavaScript, or read cluster-internal state.
"""
from __future__ import annotations

import json

ALLOWED_COLLECTION = "purchase_orders"

# Stages that write, execute arbitrary JS, or expose cluster-internal state.
# Denied recursively (including inside $facet/$lookup sub-pipelines) since a
# nested stage has the same effect as a top-level one.
DENIED_STAGES = {
    "$out",
    "$merge",
    "$function",
    "$accumulator",
    "$where",
    "$currentOp",
    "$collStats",
    "$indexStats",
    "$planCacheStats",
}

EXPORT_ROW_LIMIT = 50_000


class InvalidQueryError(ValueError):
    pass


def _check_no_denied_stages(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in DENIED_STAGES:
                raise InvalidQueryError(f"Stage '{key}' is not permitted in exports.")
            _check_no_denied_stages(value)
    elif isinstance(node, list):
        for item in node:
            _check_no_denied_stages(item)


def parse_and_validate_pipeline(command: str) -> list[dict]:
    """Parse a `db.<collection>.aggregate([...])` string into a pipeline,
    restricted to ALLOWED_COLLECTION and free of DENIED_STAGES. Raises
    InvalidQueryError on anything else — never falls back to eval()."""
    command = command.strip()
    prefix = f"db.{ALLOWED_COLLECTION}.aggregate("
    if not command.startswith(prefix):
        raise InvalidQueryError(
            f"Only 'db.{ALLOWED_COLLECTION}.aggregate([...])' commands can be exported."
        )
    if not command.endswith(")"):
        raise InvalidQueryError("Malformed command: expected a trailing ')'.")

    pipeline_text = command[len(prefix) : -1]
    try:
        pipeline = json.loads(pipeline_text)
    except json.JSONDecodeError as exc:
        raise InvalidQueryError(f"Could not parse pipeline as JSON: {exc}") from exc

    if not isinstance(pipeline, list) or not pipeline:
        raise InvalidQueryError("Pipeline must be a non-empty list of stages.")

    _check_no_denied_stages(pipeline)

    pipeline.append({"$limit": EXPORT_ROW_LIMIT})
    return pipeline
