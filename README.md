# Procurement Assistant — Backend

FastAPI + LangGraph service answering natural-language questions about California
state purchase orders (2012-2015, ~$151B in spend) via MongoDB aggregation
pipelines and semantic search.

Frontend repo: (link added here once the frontend repo is created and pushed — see Task 21)

## Quick start (no external accounts needed for local dev)

```bash
cp .env.example .env   # fill in ANTHROPIC_API_KEY and VOYAGE_API_KEY
docker-compose up --build
```

API: http://localhost:8000/health/live

## Commands

See `CLAUDE.md` for the full command reference (tests, ingestion, eval harness).
