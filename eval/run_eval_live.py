"""Runs the golden query set against the actual deployed backend over HTTP
(not an in-process run_agent() call like run_eval.py) — this exercises the
real Render process, its exact deployed code, CORS/rate-limiting, and the
live Atlas cluster, rather than just the agent logic against local
env-configured credentials.

Usage: python eval/run_eval_live.py [base_url]
Defaults to the live Render backend if no URL is given.
"""

import json
import sys
import uuid
from pathlib import Path

import httpx

# Running this as `python eval/run_eval_live.py` puts eval/ (not backend/)
# on sys.path[0], so `import eval._util` would fail without this — same
# fix run_eval.py needs for `from app.agent.graph import run_agent`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval._util import normalize_dashes  # noqa: E402

DEFAULT_BASE_URL = "https://procurement-assistant-backend.onrender.com"


def parse_sse_final_answer(response_text: str) -> str:
    final_text = ""
    for line in response_text.splitlines():
        if not line.startswith("data: "):
            continue
        chunk = json.loads(line[len("data: ") :])
        if chunk.get("type") == "final_answer":
            final_text = chunk.get("text", "")
    return final_text


def run_one(client: httpx.Client, case: dict) -> tuple[str, str]:
    # Fresh thread_id per invocation — same reasoning as run_eval.py: a
    # static id would let the checkpointer replay stale history instead of
    # genuinely re-querying the live agent.
    thread_id = f"eval-live-{case['id']}-{uuid.uuid4()}"
    response = client.post(
        "/api/chat",
        json={"message": case["question"], "conversation_id": thread_id},
        timeout=120,
    )
    response.raise_for_status()
    final_text = parse_sse_final_answer(response.text)
    if not case["expect_contains"]:
        return "review", final_text
    normalized = normalize_dashes(final_text)
    passed = any(expected in normalized for expected in case["expect_contains"])
    return ("pass" if passed else "fail"), final_text


def main() -> None:
    base_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_URL
    cases = json.loads((Path(__file__).parent / "golden_queries.json").read_text())
    results = []
    with httpx.Client(base_url=base_url) as client:
        for case in cases:
            status, answer = run_one(client, case)
            results.append(
                {"id": case["id"], "category": case["category"], "status": status, "answer": answer}
            )
            print(f"[{status.upper():6}] {case['id']}: {answer[:100]}")

    passed = sum(1 for r in results if r["status"] == "pass")
    failed = sum(1 for r in results if r["status"] == "fail")
    review = sum(1 for r in results if r["status"] == "review")
    print(
        f"\n{passed} passed, {failed} failed, {review} need manual review, out of {len(results)} total."
    )
    print(f"Target: {base_url}")

    (Path(__file__).parent / "results_live.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
