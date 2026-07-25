"""Decide whether a query's result set is worth charting, and if so, how.

Deliberately decoupled from the prose answer: the agent's own streamed text
never embeds chart data or a chart directive. Instead, after a real
mongodb_query has run for this turn, this module re-executes that same
pipeline (through the same safe, non-eval parser used by /api/export — see
app/db/safe_pipeline.py) to get a small preview of the actual rows, asks a
cheap Haiku 4.5 call to pick a chart shape from that preview, and then runs
a deterministic sanity check before ever handing anything to the frontend.
The LLM decides *if a chart would help*; plain Python decides *whether the
data can actually support one*.
"""

from __future__ import annotations

from typing import Literal

from bson import ObjectId
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.db.client import get_database
from app.db.safe_pipeline import InvalidQueryError, parse_and_validate_pipeline

HAIKU_MODEL = "claude-haiku-4-5"
CHART_PREVIEW_ROWS = 15
MAX_CHART_CATEGORIES = 12


class ChartDecision(BaseModel):
    chart_type: Literal["bar", "line", "pie", "none"] = Field(
        description="'none' unless a chart would genuinely add something the "
        "prose/table doesn't already convey — a single number or a short "
        "list of 2-3 items doesn't need one. 'line' for a trend over time "
        "(fiscal years, quarters, dates). 'pie' for a share-of-total "
        "breakdown across a handful of categories. 'bar' for ranked "
        "comparisons across categories. If the user's question explicitly "
        "asks for a chart, graph, or visualization, always pick a type — "
        "never 'none' in that case."
    )
    category_field: str | None = Field(
        default=None, description="The field name to use as the category/x-axis label."
    )
    value_field: str | None = Field(default=None, description="The numeric field name to plot.")
    title: str | None = Field(default=None, description="A short chart title, under 8 words.")


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _flatten(value: object) -> object:
    """A compound $group key (`_id: {fiscal_year, quarter}`, one of the most
    common aggregation shapes for this dataset) is a dict, not a scalar —
    left as-is, it can't be a chart category label. Flattening it into a
    single readable string (e.g. "2013-2014 3") lets the flat
    category/value schema below cover this case too, without a whole
    second multi-series schema for what's fundamentally still one axis of
    labels."""
    if isinstance(value, dict):
        return " ".join(str(v) for v in value.values())
    return value


def _sanitize_decision(decision: ChartDecision, rows: list[dict]) -> dict | None:
    """The deterministic half of this feature: an LLM saying 'bar chart on
    fields x and y' doesn't make it renderable. Reject anything that would
    produce a broken or misleading chart instead of trusting the model."""
    if decision.chart_type == "none" or not rows:
        return None
    if not decision.category_field or not decision.value_field:
        return None
    if len(rows) < 2:
        return None

    sample = rows[0]
    if decision.category_field not in sample or decision.value_field not in sample:
        return None
    if not all(_is_numeric(row.get(decision.value_field)) for row in rows):
        return None

    trimmed = rows[:MAX_CHART_CATEGORIES]
    return {
        "type": decision.chart_type,
        "title": decision.title or "",
        "category_field": decision.category_field,
        "value_field": decision.value_field,
        "data": [
            {
                "category": str(row[decision.category_field]),
                "value": row[decision.value_field],
            }
            for row in trimmed
        ],
    }


async def build_chart(user_message: str, query: str) -> dict | None:
    """Best-effort: re-runs the turn's query for a small row preview, asks
    Haiku whether/how to chart it, and returns a render-ready payload or
    None. Never raises — a failure here should never affect the real
    answer, only mean no chart is attached."""
    try:
        pipeline = parse_and_validate_pipeline(query)
    except InvalidQueryError:
        return None

    try:
        db = get_database()
        rows = await db.purchase_orders.aggregate(pipeline).to_list(length=CHART_PREVIEW_ROWS)
    except Exception:  # noqa: BLE001 — best-effort, see docstring
        return None

    if not rows:
        return None

    # Only drop `_id` when it's a real BSON ObjectId (raw-document lookups,
    # rare for this agent's pipelines) — in a $group aggregation, which is
    # how almost every one of this agent's queries gets its data, `_id` IS
    # the group key (e.g. a department name) and is exactly the category
    # field a chart needs. Stripping it unconditionally silently made
    # charting impossible for the most common query shape.
    preview_rows = [
        {k: _flatten(v) for k, v in row.items() if not isinstance(v, ObjectId)} for row in rows
    ]
    if not preview_rows or not preview_rows[0]:
        return None

    try:
        settings = get_settings()
        llm = ChatAnthropic(model=HAIKU_MODEL, api_key=settings.anthropic_api_key, max_tokens=300)
        structured = llm.with_structured_output(ChartDecision)
        decision = await structured.ainvoke(
            "A user asked a California state procurement question and this data "
            "was retrieved to answer it. Decide whether a chart would help, and "
            "if so, what kind and which fields to plot.\n\n"
            f"User's question: {user_message}\n\n"
            f"Result rows (preview): {preview_rows}"
        )
    except Exception:  # noqa: BLE001 — best-effort, see docstring
        return None

    return _sanitize_decision(decision, preview_rows)
