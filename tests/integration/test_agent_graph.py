# backend/tests/integration/test_agent_graph.py
import pytest

from app.agent.graph import run_agent


@pytest.mark.asyncio
async def test_run_agent_answers_total_orders_question():
    """Requires a seeded MongoDB (Task 6) and a real ANTHROPIC_API_KEY —
    this is an integration test against the real stack, not a mock."""
    chunks = []
    async for chunk in run_agent(
        "How many purchase orders were created in fiscal year 2013-2014?",
        thread_id="test-thread-1",
    ):
        chunks.append(chunk)

    final_answers = [c for c in chunks if c.get("type") == "final_answer"]
    assert len(final_answers) == 1
    assert "120,636" in final_answers[0]["text"] or "120636" in final_answers[0]["text"]
