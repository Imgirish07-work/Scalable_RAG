# Scalable RAG — Backend

FastAPI layer wrapping the RAG pipeline.

> **Auth status:** Removed in Phase 0 to unblock a clean endpoint rebuild.
> Every endpoint is currently unauthenticated. Auth (JWT + API keys) returns in Phase 8 — see [PLAN.md](../PLAN.md).

## Endpoints

| Method | Path              | Purpose                                  |
|--------|-------------------|------------------------------------------|
| GET    | `/healthz`        | Liveness                                 |
| GET    | `/readyz`         | Readiness (waits for `pipeline.initialize()`) |
| GET    | `/metrics`        | Prometheus exposition                    |
| POST   | `/v1/query`       | Synchronous RAG query                    |
| POST   | `/v1/ingest`      | Multipart upload + ingest                |
| GET    | `/v1/collections` | List configured collections              |

## Quickstart

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY or GEMINI_API_KEY.

docker compose --profile dev up --build

curl -X POST http://localhost:8000/v1/ingest `
  -F "file=@./data/sample_docs/your-file.pdf" `
  -F "collection=my-docs"

curl -X POST http://localhost:8000/v1/query `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"summarize this\",\"collection\":\"my-docs\",\"top_k\":5}'
```

## Layout (MVCR + Service layer)

```
backend/
├── main.py                 FastAPI app factory + lifespan
├── settings.py             BackendSettings (env-driven config)
├── dependencies.py         FastAPI Depends providers (get_pipeline)
├── middleware.py           HTTP middleware (request-id, access log)
├── metrics.py              Prometheus counters and histograms
│
├── api/v1/                 Controllers — HTTP route handlers
│   ├── health.py
│   ├── query.py
│   ├── documents.py        (POST /v1/ingest today; /v1/documents in Phase 2)
│   └── collections.py
│
├── schemas/                Pydantic request + response DTOs
│   ├── common.py
│   ├── query.py            QueryRequest
│   ├── document.py         DocumentCreatedView
│   └── collection.py       CollectionView, CollectionListView
│
├── models/                 SQLAlchemy ORM entities (populated in Phase 2+)
│   └── base.py             class Base(DeclarativeBase)
│
├── repositories/           Data access (only layer that writes ORM queries)
│   └── database.py         engine, session factory, session_scope
│
├── services/               Business logic (populated in Phase 2+)
└── db/                     SQL migrations + runner (populated in Phase 2+)
```

## Operational notes

- First boot: ~30-90 s (pip install + pipeline warmup). Watch for `Backend ready in N ms`.
- `/readyz` returns 503 until warmup completes.
- Set `RERANKER_ENABLED=false` if cross-encoder model files aren't available.
- All requests currently run as the hardcoded `dev-user` while auth is removed.

## Class-naming conventions inside `schemas/`

| Suffix         | Direction | Example                       |
|----------------|-----------|-------------------------------|
| `*Request`     | Inbound   | `QueryRequest`                |
| `*Filter`      | Inbound   | (Phase 4) `DocumentFilter`    |
| `*View`        | Outbound  | `CollectionView`              |
| `*ListView`    | Outbound  | `CollectionListView`          |
| `*CreatedView` | Outbound  | `DocumentCreatedView`         |
