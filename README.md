# CA Procurement Assistant — Backend

A FastAPI + LangGraph service that answers natural-language questions about
California state purchase orders (fiscal years 2012–2015, 346,018 records,
~$151B in spend) by generating and running MongoDB aggregation pipelines,
with a semantic-search layer over UNSPSC purchasing categories for
fuzzy/conceptual questions the exact data wording won't match.

**Live API:** https://procurement-assistant-backend.onrender.com
**Frontend repo:** https://github.com/halsabbah10/procurement-assistant-frontend
**Live app:** https://frontend-two-mu-97.vercel.app

> First request after ~15 minutes of inactivity takes 30–60s — Render's free
> tier spins the instance down when idle. The frontend shows a "waking up"
> state for this.

## How it works

```
                         ┌─────────────────────────────┐
  React frontend  ──SSE──▶  FastAPI (/api/chat)         │
  (Vercel)                │                              │
                         │  1. Haiku 4.5 out-of-scope    │
                         │     guard (context-aware)     │
                         │  2. LangGraph ReAct agent      │
                         │     on Sonnet 5                │──▶ MongoDB Atlas
                         │     - mongodb_query /           │    (procurement db:
                         │       mongodb_schema /          │     purchase_orders,
                         │       mongodb_query_checker      │     item_embeddings)
                         │       (langchain-mongodb        │
                         │       toolkit)                  │──▶ checkpointing_db
                         │     - semantic_search           │    (LangGraph state +
                         │       (Voyage embeddings +      │     conversation list)
                         │       $vectorSearch)            │
                         │  3. Escalate to Opus 4.8 once   │──▶ Voyage AI
                         │     if Sonnet can't finish       │    (voyage-4, 512-dim
                         │  4. Best-effort enrichment:      │     embeddings)
                         │     query transparency,          │
                         │     follow-up suggestions,       │──▶ Anthropic API
                         │     auto-generated chart         │    (Sonnet 5 / Opus 4.8
                         │     (all via separate Haiku      │     / Haiku 4.5)
                         │     calls, run concurrently)     │
                         └─────────────────────────────┘
```

Every structured answer is grounded in a real MongoDB aggregation the model
generated and ran — the UI shows that exact query (`GET`-able again via
`/api/export` for CSV/JSON download). Semantic search doesn't answer
questions directly; it resolves a fuzzy phrase ("cybersecurity spending")
to real `commodity_title`/`class_title` values, which are then used to
filter a normal structured aggregation — so numeric answers are always
exact, never an LLM's approximation from embedding similarity.

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | async-native, matches Motor's async driver |
| Agent orchestration | LangGraph (`create_react_agent`) | prebuilt ReAct loop + first-class MongoDB checkpointing |
| LLM | Claude, tiered: Haiku 4.5 (routing/guard/enrichment) → Sonnet 5 (primary) → Opus 4.8 (escalation only) | cost/latency scaled to task difficulty, not one flagship model everywhere |
| Structured queries | `langchain-mongodb`'s `MongoDBDatabaseToolkit` | text-to-MQL toolkit purpose-built for this |
| Semantic search | Voyage AI `voyage-4` embeddings (512-dim) + MongoDB `$vectorSearch` | Matryoshka-trained — dimension cut verified empirically, not guessed (see Design decisions) |
| Database | MongoDB Atlas (M0 free tier) / `mongodb-atlas-local` for offline dev | same `$vectorSearch` support locally as in the cloud |
| Deployment | Render (Blueprint, `render.yaml`) | free tier, straightforward FastAPI support |

## Repo layout

```
app/
├── main.py                # FastAPI app, CORS, router registration, global exception handler
├── core/
│   ├── config.py          # pydantic-settings env config
│   ├── logging.py         # structlog JSON logging setup
│   ├── rate_limit.py      # in-memory per-IP + daily-cap limiter
│   └── ttl_cache.py       # in-memory TTL cache (analytics summary)
├── db/
│   ├── client.py          # Motor client; get_database() (procurement) / get_checkpoint_database() (checkpointing_db)
│   ├── ingest.py          # pure CSV-row parsing functions, unit-tested in isolation
│   ├── embeddings.py      # dedupe_items(): rows -> distinct UNSPSC category hierarchies
│   ├── conversations.py   # touch_conversation(): sidebar metadata upsert
│   └── safe_pipeline.py   # non-eval, allowlisted query parser for /api/export
├── agent/
│   ├── graph.py           # run_agent(): the outer orchestrator (see below)
│   ├── tools.py           # structured MongoDB tools + semantic_search tool
│   ├── prompts.py         # system prompt
│   ├── chart_spec.py      # best-effort chart-suggestion (separate Haiku call + deterministic sanity gate)
│   └── state.py           # LangGraph state schema
├── routers/
│   ├── chat.py            # POST /api/chat (SSE) — the only endpoint that calls an LLM
│   ├── analytics.py       # GET /api/analytics/summary, /api/analytics/department/{name}
│   ├── conversations.py   # conversation list/rename/delete/history
│   ├── export.py          # POST /api/export (CSV/JSON)
│   └── health.py          # liveness/readiness
└── schemas/                # Pydantic request models

scripts/
├── run_ingestion.py        # CSV -> purchase_orders (idempotent)
└── run_embeddings.py       # purchase_orders -> item_embeddings + vector index (idempotent)

eval/
├── golden_queries.json     # 8 queries across the required categories + edge cases
├── run_eval.py             # runs them via an in-process run_agent() call (local/CI agent-logic check)
└── run_eval_live.py        # runs them over HTTP against a real deployed backend

tests/
├── unit/                   # pure functions — no DB, no network
└── integration/            # against a real MongoDB (local or CI service container), no mocks
```

## Local setup

No external accounts needed — `mongodb-atlas-local` gives full `$vectorSearch`
parity offline.

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY (both free-tier)
docker-compose up --build
python scripts/run_ingestion.py    # one-time: load the CSV into MongoDB
python scripts/run_embeddings.py   # one-time: generate + index category embeddings
```

API: http://localhost:8000/health/live

### Environment variables

| Variable | Local default | Notes |
|---|---|---|
| `MONGODB_URI` | `mongodb://mongodb:27017` (docker-compose) | swap for an Atlas SRV string in production |
| `MONGODB_DB_NAME` | `procurement` | |
| `ANTHROPIC_API_KEY` | — | console.anthropic.com — programmatic key, separate from any Claude Code login |
| `VOYAGE_API_KEY` | — | dashboard.voyageai.com — free tier covers this project many times over |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | comma-separated, CORS allowlist |
| `RATE_LIMIT_PER_MINUTE` | `20` | per-IP |
| `DAILY_REQUEST_CAP` | `200` | whole-service circuit breaker |

## Commands

```bash
pip install -e ".[dev]"              # install (app + dev deps)
uvicorn app.main:app --reload        # dev server

pytest                                # unit + integration tests
ruff check .                          # lint
black --check .                       # format check (black . to apply)

python scripts/run_ingestion.py       # CSV -> MongoDB (idempotent)
python scripts/run_embeddings.py      # embeddings + vector index (idempotent)

python eval/run_eval.py               # golden queries against local agent logic
python eval/run_eval_live.py [url]    # golden queries against a real deployed backend (default: production)
```

## API reference

All endpoints are under the Render URL above in production, `localhost:8000` locally.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/chat` | SSE stream of agent steps + a final answer enriched with `query`, `suggestions`, and `chart` fields |
| `GET` | `/api/analytics/summary` | Fiscal-year/quarterly spend, acquisition-type breakdown, top departments/suppliers (TTL-cached, 5 min) |
| `GET` | `/api/analytics/department/{name}` | Drill-down: totals, spend by year, top suppliers/categories for one department |
| `GET` | `/api/conversations` | List conversations (sidebar), sorted by most recently updated |
| `PATCH` | `/api/conversations/{id}` | Rename |
| `DELETE` | `/api/conversations/{id}` | Soft-delete |
| `GET` | `/api/conversations/{id}/messages` | Reconstructs human/assistant turns from the LangGraph checkpoint (text only — see Design decisions) |
| `POST` | `/api/export` | Re-runs a `db.purchase_orders.aggregate([...])` query (validated, not eval'd — see Design decisions) and returns CSV or JSON |
| `GET` | `/health/live` | Liveness (no DB call) |
| `GET` | `/health/ready` | Readiness (pings MongoDB) |

`POST /api/chat` request body: `{"message": string, "conversation_id": string}`.
SSE `data:` lines are one of:
```jsonc
{"type": "step", "text": "..."}                                          // intermediate progress
{"type": "final_answer", "text": "...", "query"?: "...", "suggestions"?: [...], "chart"?: {...}}
{"type": "error", "text": "..."}
```

## Design decisions

**Tiered Claude models, not one flagship everywhere.** Haiku 4.5 handles the
out-of-scope guard and best-effort enrichment (follow-ups, chart
suggestion) — cheap, fast, and the task doesn't need more. Sonnet 5 is the
default agent model. Opus 4.8 is a rare escalation, only when Sonnet
exhausts its step budget without producing an answer (~0% of runs after
the fixes below, occasionally more on genuinely hard multi-hop questions).

**Semantic layer embeds UNSPSC categories, not raw item text.** The
original design (distinct `item_name`/`item_description` text, 246,859
rows) didn't fit MongoDB Atlas's free-tier 512MB storage quota alongside
`purchase_orders`. Redesigned to embed the UNSPSC category hierarchy
instead (segment > family > class > commodity — 13,294 distinct
combinations) — a genuine quality improvement (category-level matching is
more robust to item-text wording variance), not just a storage hack. Still
didn't fit at 1024 dimensions; cut to 512 via Voyage's `output_dimension`
parameter (`voyage-4` is Matryoshka-trained, so this needs no
re-embedding). Verified empirically before committing: 8 representative
queries against a real 1024-dim vs. 512-dim collection built from this
exact dataset, averaging 4.4/5 top-5 overlap with the same top-1 hit in
6/8 cases — negligible quality loss for a filter-then-aggregate use case.

**The export endpoint never uses `eval()`.** `langchain-mongodb`'s own
query parser runs `eval(agg_str, {"ObjectId": ..., "datetime": ...})` —
omitting `__builtins__` from that globals dict does not sandbox eval();
Python auto-injects the real one when the key is absent, so that call is
one `__import__` away from arbitrary code execution. Acceptable as an
internal detail when the input is the model's own tool-call argument;
never safe to expose to a client-supplied string, which is exactly what
"re-run the query shown in the transparency panel" is. `app/db/safe_pipeline.py`
instead uses `json.loads` only (structurally incapable of executing code),
a hard-coded single-collection allowlist, and a recursive denylist of
write/JS-exec/introspection aggregation stages (`$out`, `$merge`,
`$function`, `$where`, `$collStats`, etc.).

**Conversation history is reconstructed from the LangGraph checkpoint, not
a separate store.** `checkpointing_db.conversations` holds only sidebar
metadata (title, timestamps); `GET /api/conversations/{id}/messages`
rebuilds the actual human/assistant turns straight from the same
checkpoint the agent uses for its own memory, so reopening a conversation
is backed by the same durable record — not a second, potentially
inconsistent source of truth. The trade-off: query/suggestions/chart
enrichments aren't part of the checkpoint (they're regenerated per turn,
not conversational state), so a conversation reopened on a different
browser shows prose only. The frontend's localStorage cache carries full
fidelity for the common case (same browser).

**No client-embedded API key or auth gate.** A secret shipped in public SPA
JS isn't real protection. Rate limiting (per-IP + a whole-service daily
cap) plus CORS is the actual mitigation for a public demo endpoint with no
multi-user requirement.

## Testing strategy

- **Unit** (`tests/unit/`): pure functions only — CSV parsing, dedup logic,
  chart sanity-gate, TTL cache — no DB, no network, runs in ~1s.
- **Integration** (`tests/integration/`): against a real MongoDB (local
  Docker or a CI service container), never mocked. `test_agent_graph.py`
  (a real end-to-end agent question) is excluded from CI specifically
  because it needs the full 346,018-row dataset seeded, which CI's empty
  service container doesn't have — it remains a real, valuable local/live
  check.
- **Eval harness** (`eval/`): 8 golden queries spanning the assessment's
  required categories plus edge cases (out-of-scope, multi-step
  aggregation, semantic fuzzy-matching). `run_eval.py` checks agent logic
  in-process; `run_eval_live.py` hits the actual deployed HTTP endpoint —
  the latter is what should be re-run after any deploy to confirm the real
  running service, not just the code, behaves correctly.
- **E2E**: lives in the frontend repo (Playwright), since it exercises the
  full browser → API → DB round trip.

## Known limitations

- Single-instance rate limiting and TTL cache (in-memory) — would need
  Redis if this ever ran on more than one Render instance.
- No multi-user auth/RBAC — out of scope for a single public demo
  deployment with no per-user data.
- `POST /api/export` caps results at 50,000 rows.
