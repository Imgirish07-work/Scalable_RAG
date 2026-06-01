# Scalable RAG — Backend

FastAPI layer wrapping the RAG pipeline.

> **Auth status:** Removed in Phase 0 to unblock a clean endpoint rebuild.
> Every endpoint is currently unauthenticated. Auth (JWT + API keys) returns in Phase 8 — see [PLAN.md](../PLAN.md).
>
> **Phase 3 (Async Ingestion) is complete.** Document uploads are now non-blocking: the API enqueues an ARQ job and returns 202; a separate worker container handles the ingestion pipeline. See the workers/ and services/ sections below.

## Endpoints

| Method | Path              | Purpose                                  |
|--------|-------------------|------------------------------------------|
| GET    | `/healthz`        | Liveness                                 |
| GET    | `/readyz`         | Readiness (waits for `pipeline.initialize()`) |
| GET    | `/metrics`        | Prometheus exposition                    |
| POST   | `/v1/query`       | Synchronous RAG query                    |
| POST   | `/v1/ingest`      | Multipart upload → enqueues async job, returns 202 + `{ job_id, document_id }` |
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
│                           (ingest_jobs_queued_total, ingest_jobs_inflight, ingest_jobs_failed_total)
│
├── api/v1/                 Controllers — HTTP route handlers
│   ├── health.py
│   ├── query.py
│   ├── documents.py        POST /v1/ingest → enqueues ARQ job, returns 202
│   └── collections.py
│
├── schemas/                Pydantic request + response DTOs
│   ├── common.py
│   ├── query.py            QueryRequest
│   ├── document.py         DocumentCreatedView
│   └── collection.py       CollectionView, CollectionListView
│
├── models/                 SQLAlchemy ORM entities
│   └── base.py             class Base(DeclarativeBase)
│
├── repositories/           Data access (only layer that writes ORM queries)
│   └── database.py         engine, session factory, session_scope
│
├── services/               Business logic (Phase 3 additions)
│   ├── ingestion_service.py   Orchestrates chunk → embed → upsert with per-chunk progress callbacks
│   ├── redis_event_bus.py     Redis pub/sub publisher; backend pods subscribe + SSE fan-out
│   ├── pipeline_factory.py    Lazy pipeline singleton shared across worker tasks
│   └── orphan_sweeper.py      Periodic task: detects stuck jobs via processing_started_at lease
│
├── workers/                ARQ async job workers (run in separate Docker container)
│   ├── arq_settings.py        WorkerSettings — Redis pool, max_jobs, timeouts
│   ├── queue_client.py        enqueue_ingest_job() helper used by API handlers
│   └── tasks.py               ingest_document_task — claims job, runs IngestionService, updates status
│
└── db/                     SQL migrations
    └── versions/
        └── 003_processing_started_at.py   adds processing_started_at column + index on documents
```

## Operational notes

- First boot: ~30-90 s (pip install + pipeline warmup). Watch for `Backend ready in N ms`.
- `/readyz` returns 503 until warmup completes.
- Set `RERANKER_ENABLED=false` if cross-encoder model files aren't available.
- All requests currently run as the hardcoded `dev-user` while auth is removed.
- The worker container must share the same Redis instance as the API. Set `REDIS_URL` in both services.
- `OrphanSweeper` runs inside the worker process. If only the API is running (no worker), stuck jobs will remain in `processing` status until a worker starts.
- Ingestion progress SSE events are published on the Redis channel `ingest:progress:{doc_id}`. Clients that connect after ingestion completes will miss intermediate events; they should fall back to polling document status.
- Migration 003 must be applied before starting the Phase 3 worker (`alembic upgrade head`).

## Class-naming conventions inside `schemas/`

| Suffix         | Direction | Example                       |
|----------------|-----------|-------------------------------|
| `*Request`     | Inbound   | `QueryRequest`                |
| `*Filter`      | Inbound   | (Phase 4) `DocumentFilter`    |
| `*View`        | Outbound  | `CollectionView`              |
| `*ListView`    | Outbound  | `CollectionListView`          |
| `*CreatedView` | Outbound  | `DocumentCreatedView`         |
