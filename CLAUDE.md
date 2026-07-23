# Backend — Procurement Assistant API

FastAPI service exposing a LangGraph agent over SSE, backed by MongoDB.

## Commands
- `pip install -e ".[dev]"` — install
- `uvicorn app.main:app --reload` — run dev server
- `pytest` — unit + integration tests
- `python scripts/run_ingestion.py` — load the CSV into MongoDB (idempotent, safe to re-run)
- `python scripts/run_embeddings.py` — generate item embeddings (run after ingestion)
- `python eval/run_eval.py` — run the golden-query eval harness

## Layout
- `app/db/ingest.py` — pure parsing functions (price, qualifications, location, dates, quarter). Unit-tested in isolation; the ingestion script is a thin orchestrator over these.
- `app/agent/` — LangGraph agent. `tools.py` wires up `MongoDBDatabaseToolkit` (structured queries) and the Voyage-backed semantic search tool; `graph.py` is the outer orchestrator that tries Sonnet 5 first and escalates to Opus 4.8 on repeated failure.
- `app/routers/chat.py` — the only endpoint that touches the LLM; everything else (`health.py`, `analytics.py`) is plain MongoDB aggregation, no LLM call.

## Gotchas
- `MongoDBSaver` (LangGraph checkpointer) wraps a **sync** `pymongo.MongoClient`, not Motor. It's called from async route handlers via `asyncio.to_thread()` — see `app/agent/graph.py`. If a newer package version ships an async variant, prefer it, but don't assume one exists without checking `python -c "from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver"` first.
- `Supplier Qualifications` in the source CSV is space-separated, not comma-separated — see the note in `app/db/ingest.py::parse_qualifications`.
