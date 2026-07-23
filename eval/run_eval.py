# backend/eval/run_eval.py
"""Runs the golden query set against the live agent and reports pass/fail.
Queries with an empty expect_contains are graded manually (open-ended) —
this script reports them as "review" rather than pass/fail.

Usage: python eval/run_eval.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.graph import run_agent  # noqa: E402


async def run_one(case: dict) -> tuple[str, str]:
    final_text = ""
    async for chunk in run_agent(case["question"], thread_id=f"eval-{case['id']}"):
        if chunk["type"] == "final_answer":
            final_text = chunk["text"]
    if not case["expect_contains"]:
        return "review", final_text
    passed = any(expected in final_text for expected in case["expect_contains"])
    return ("pass" if passed else "fail"), final_text


async def main() -> None:
    cases = json.loads((Path(__file__).parent / "golden_queries.json").read_text())
    results = []
    for case in cases:
        status, answer = await run_one(case)
        results.append({"id": case["id"], "category": case["category"], "status": status, "answer": answer})
        print(f"[{status.upper():6}] {case['id']}: {answer[:100]}")

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    review = sum(1 for r in results if r["status"] == "review")
    print(f"\n{passed} passed, {failed} failed, {review} need manual review, out of {len(results)} total.")

    (Path(__file__).parent / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
