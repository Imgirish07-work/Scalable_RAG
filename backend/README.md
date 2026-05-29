# Scalable RAG — Backend

FastAPI layer wrapping the RAG pipeline.

> **Auth status:** Removed in Phase 0 to unblock a clean endpoint rebuild.
> Every endpoint is currently unauthenticated. Auth (JWT + API keys) returns in Phase 8 — see [PLAN.md](../PLAN.md).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET  | `/healthz`        | Liveness |
| GET  | `/readyz`         | Readiness (waits for pipeline.initialize()) |
| GET  | `/metrics`        | Prometheus exposition |
| POST | `/v1/query`       | Synchronous RAG query |
| POST | `/v1/ingest`      | Multipart upload + ingest |
| GET  | `/v1/collections` | List configured collections |

## Quickstart

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY or GEMINI_API_KEY.

docker compose --profile dev up --build

# Ingest
curl -X POST http://localhost:8000/v1/ingest `
  -F "file=@./data/sample_docs/your-file.pdf" `
  -F "collection=my-docs"

# Query
curl -X POST http://localhost:8000/v1/query `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"summarize this\",\"collection\":\"my-docs\",\"top_k\":5}'
```

## Layout

```
backend/
├── main.py            FastAPI app + lifespan
├── config.py          BackendSettings
├── deps.py            get_pipeline
├── middleware/        request_id
├── routers/           health, query, ingest, collections
├── repos/             Async SQLAlchemy base (dormant until Phase 2)
├── models/            Pydantic API shapes
├── observability/     Prometheus metrics
└── migrations/        Alembic (no migrations yet — Phase 2 adds the first one)
```

## Operational notes

- First boot: ~30-90 s (pip install + pipeline warmup). Watch for `Backend ready in N ms`.
- `/readyz` returns 503 until warmup completes.
- Set `RERANKER_ENABLED=false` if cross-encoder model files aren't available.
- All requests run as the hardcoded `dev-user` while auth is removed.
