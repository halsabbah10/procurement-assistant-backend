"""Safely parse and validate a `db.<collection>.aggregate([...])` command
string, without ever using `eval()`.

Three callers share this module: the agent's own `mongodb_query` tool
(`app/agent/tools.py`), the chart-generation re-execution of that same query
(`app/agent/chart_spec.py`), and the CSV/JSON export endpoint
(`app/routers/export.py`). All three used to go through — or, in the
agent's case, still went through until this fix — `langchain_mongodb`'s
`MongoDBDatabase._parse_command`, which parses the pipeline with
`eval(agg_str, {"ObjectId": ..., "datetime": ..., "timezone": ...})`. That
call omits `__builtins__` from the globals dict, which does NOT sandbox
eval() — Python auto-injects the real `__builtins__` into any globals dict
that doesn't already have the key, so that code path has access to
`__import__`, `open`, `exec`, etc. For the export endpoint this was always
a client-reachable RCE surface if ever wired up naively; for the agent's
own query tool it's reachable via ordinary prompt injection in a chat
message, since the `query` tool-call argument is LLM output steered by
untrusted conversation text, not a trusted internal value.

This module uses only `ast.literal_eval` (structurally incapable of
executing code, calling functions, or referencing names other than
True/False/None) plus a small sentinel-based pre/post-processing step for
the handful of JS-style MongoDB constructs (`ISODate(...)`, `new
Date(...)`, `ObjectId(...)`, a bare unquoted `_id` key) that real generated
queries use and that plain JSON can't express. This is intentionally MORE
tolerant than a strict `json.loads`-only parser (which silently failed to
parse a large share of real queries — see chart_spec.py's docstring) while
being exactly as safe as one: no additional Python syntax is accepted
beyond literal containers and the four specific constructs handled below.
"""

from __future__ import annotations

import ast
import re
from datetime import datetime

from bson import ObjectId

# Stages that write, execute arbitrary JS, expose cluster-internal state,
# join across collections, or reshape many input documents into one output
# document with unbounded nested arrays. Denied everywhere (agent queries,
# chart re-execution, export), not just export, because:
#   - $lookup/$graphLookup/$unionWith: a DoS vector via an uncorrelated
#     sub-pipeline re-run once per outer document — this app's architecture
#     never needs a cross-collection join (semantic search and structured
#     aggregation are deliberately separate, sequential tool calls).
#   - $facet/$bucket/$bucketAuto: their one-document-with-nested-arrays
#     shape defeats size bounds that assume "N output documents" (the
#     export row limit below, and — for the agent/chart paths — whatever
#     is iterating the result afterward). Not needed anywhere in this app's
#     documented query pattern (each aggregation is one simple, focused
#     pipeline; multi-metric questions are multiple sequential tool calls).
# Denied recursively (including inside nested sub-pipelines) since a nested
# stage has the same effect as a top-level one.
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
    "$lookup",
    "$graphLookup",
    "$unionWith",
    "$facet",
    "$bucket",
    "$bucketAuto",
}

EXPORT_ROW_LIMIT = 50_000

# Generous ceiling on server-side aggregation time, applied by callers via
# maxTimeMS — bounds worst-case cost of any pipeline that gets past the
# denylist (e.g. an expensive but structurally-permitted $match/$sort over
# a large slice of the collection).
MAX_QUERY_TIME_MS = 20_000


class InvalidQueryError(ValueError):
    pass


# --- JS-style construct -> safe sentinel conversion -------------------------
#
# Real generated queries commonly use MongoDB-shell conventions plain JSON
# can't express: ISODate(...)/new Date(...)/ObjectId(...) constructor calls,
# and a bare (unquoted) `_id` key — extremely common since `_id` is the
# group key in the large majority of this app's $group aggregations. Each
# construct is rewritten into an inert string-literal sentinel *before*
# parsing (so ast.literal_eval never sees a function call, only a string),
# then converted to the real typed value *after* parsing.

_ISO_DATE_RE = re.compile(r'ISODate\(\s*["\']([^"\']*)["\']\s*\)')
_NEW_DATE_RE = re.compile(r'new\s+Date\(\s*["\']([^"\']*)["\']\s*\)')
_OBJECT_ID_RE = re.compile(r'ObjectId\(\s*["\']([^"\']*)["\']\s*\)')
_BARE_ID_KEY_RE = re.compile(r'(?<!["\'])\b_id\b(?!["\'])')

# Not a null byte or other control character: Python's own tokenizer
# rejects null bytes in source text outright ("source code string cannot
# contain null bytes"), which would make ast.literal_eval fail on every
# query that needs a date/id conversion. Collision with real data is not a
# security concern even so — see module docstring: this text is the LLM's
# own generated query, not attacker-supplied data flowing through it, and a
# collision would at worst misinterpret one string as a date/id, never
# execute anything.
_DATE_SENTINEL_PREFIX = "MPS_DATE"
_OID_SENTINEL_PREFIX = "MPS_OID"


def _to_date_sentinel(match: re.Match) -> str:
    return f'"{_DATE_SENTINEL_PREFIX}{match.group(1)}"'


def _to_oid_sentinel(match: re.Match) -> str:
    return f'"{_OID_SENTINEL_PREFIX}{match.group(1)}"'


def _prepare_literal_text(agg_str: str) -> str:
    agg_str = _ISO_DATE_RE.sub(_to_date_sentinel, agg_str)
    agg_str = _NEW_DATE_RE.sub(_to_date_sentinel, agg_str)
    agg_str = _OBJECT_ID_RE.sub(_to_oid_sentinel, agg_str)
    agg_str = _BARE_ID_KEY_RE.sub('"_id"', agg_str)
    return agg_str


def _coerce_sentinels(node: object) -> object:
    if isinstance(node, dict):
        return {k: _coerce_sentinels(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_coerce_sentinels(v) for v in node]
    if isinstance(node, str):
        if node.startswith(_DATE_SENTINEL_PREFIX):
            raw = node[len(_DATE_SENTINEL_PREFIX) :]
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError as exc:
                raise InvalidQueryError(f"Invalid date value {raw!r}: {exc}") from exc
        if node.startswith(_OID_SENTINEL_PREFIX):
            raw = node[len(_OID_SENTINEL_PREFIX) :]
            try:
                return ObjectId(raw)
            except Exception as exc:  # noqa: BLE001 — bson raises its own InvalidId
                raise InvalidQueryError(f"Invalid ObjectId value {raw!r}: {exc}") from exc
    return node


def _parse_literal_pipeline(pipeline_text: str) -> list:
    """Parse a MongoDB-shell-flavored aggregation pipeline string into a
    Python list of stage dicts, without eval(). Tolerates the same input
    shapes the previous eval()-based parser did (Python True/False/None,
    single- or double-quoted strings, ISODate/new Date/ObjectId, a bare
    `_id` key) while being structurally incapable of executing code:
    ast.literal_eval only ever constructs literal containers and constants,
    never calls a function or resolves an arbitrary name."""
    prepared = _prepare_literal_text(pipeline_text)
    try:
        pipeline = ast.literal_eval(prepared)
    except (ValueError, SyntaxError, TypeError) as exc:
        raise InvalidQueryError(f"Could not parse aggregation pipeline: {exc}") from exc
    if not isinstance(pipeline, list) or not pipeline:
        raise InvalidQueryError("Pipeline must be a non-empty list of stages.")
    return _coerce_sentinels(pipeline)


def _check_no_denied_stages(node: object) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in DENIED_STAGES:
                raise InvalidQueryError(f"Stage '{key}' is not permitted.")
            _check_no_denied_stages(value)
    elif isinstance(node, list):
        for item in node:
            _check_no_denied_stages(item)


def _split_command(command: str) -> tuple[str, str]:
    """Split `db.<collection>.aggregate([...])` into (collection, pipeline_text)."""
    command = command.strip()
    if not command.startswith("db."):
        raise InvalidQueryError("Command must start with 'db.<collection>.aggregate('.")
    if not command.endswith(")"):
        raise InvalidQueryError("Malformed command: expected a trailing ')'.")
    try:
        rest = command[3:]
        collection, _, tail = rest.partition(".aggregate(")
        if not tail:
            raise ValueError("missing '.aggregate('")
        pipeline_text = tail[:-1]  # strip trailing ')'
    except Exception as exc:
        raise InvalidQueryError(f"Could not parse command: {exc}") from exc
    if not collection:
        raise InvalidQueryError("Could not extract a collection name.")
    return collection, pipeline_text


def parse_agent_pipeline(command: str, allowed_collections: set[str]) -> tuple[str, list]:
    """Parse a pipeline generated by the agent's own `mongodb_query` tool
    call. Restricted to `allowed_collections` (the real collections in this
    app's database) and free of DENIED_STAGES. Returns (collection, pipeline).
    Never falls back to eval()."""
    collection, pipeline_text = _split_command(command)
    if collection not in allowed_collections:
        raise InvalidQueryError(f"Collection '{collection}' does not exist.")
    pipeline = _parse_literal_pipeline(pipeline_text)
    _check_no_denied_stages(pipeline)
    return collection, pipeline


def parse_and_validate_pipeline(command: str, allowed_collections: set[str]) -> tuple[str, list]:
    """Parse a pipeline for re-execution outside the agent (chart preview
    re-run, CSV/JSON export). Same safety guarantees as
    `parse_agent_pipeline`, plus a forced trailing $limit specifically to
    bound a client-reachable export's result size."""
    collection, pipeline = parse_agent_pipeline(command, allowed_collections)
    pipeline.append({"$limit": EXPORT_ROW_LIMIT})
    return collection, pipeline
