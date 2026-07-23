# backend/tests/integration/test_agent_graph.py
import uuid

import pytest

from app.agent.graph import run_agent


@pytest.mark.asyncio
async def test_run_agent_answers_total_orders_question():
    """Requires a seeded MongoDB (Task 6) and a real ANTHROPIC_API_KEY —
    this is an integration test against the real stack, not a mock."""
    # NOTE: a fresh thread_id per run, not the brief's hardcoded
    # "test-thread-1". A shared, never-cleared thread_id lets checkpointed
    # state from a prior run accumulate on the same LangGraph checkpointer
    # thread — a rerun can then be served straight from that prior
    # checkpoint (answering in ~3-5s) instead of genuinely exercising the
    # live agent loop (~17s+), which silently defeats this as a real
    # regression test. A unique thread_id makes every run a genuinely fresh
    # conversation, with no cleanup step required.
    chunks = []
    async for chunk in run_agent(
        "How many purchase orders were created in fiscal year 2013-2014?",
        thread_id=str(uuid.uuid4()),
    ):
        chunks.append(chunk)

    final_answers = [c for c in chunks if c.get("type") == "final_answer"]
    assert len(final_answers) == 1
    assert "120,636" in final_answers[0]["text"] or "120636" in final_answers[0]["text"]
