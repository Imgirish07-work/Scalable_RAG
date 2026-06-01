# Scalable RAG — Production Plan

**Target:** Production-grade RAG backend supporting 1,000–2,000 concurrent users.
**Author:** Drafted 2026-05-29.
**Status:** Phase 0–3 complete. Phase 4+ is planned — review and revise before execution.

---

## 0. North-Star Goals

1. **Stateless, horizontally scalable HTTP layer** — any backend pod serves any request.
2. **Async-first ingestion** — uploads return a job ID immediately; heavy work runs in workers.
3. **Read-optimized data plane** — Qdrant for vectors, Postgres + Redis cache for metadata.
4. **S3-backed object storage** for raw files (MinIO locally, swap-compatible with AWS S3).
5. **Production API hygiene** — versioning, pagination, RFC 7807 errors, idempotency, rate limits.
6. **Observable** — structured logs, Prometheus metrics, OpenTelemetry traces, SLOs.
7. **Resilient** — circuit breakers, timeouts, bulkheads, graceful shutdown, DLQ.
8. **Auth removable now, re-addable cleanly later** — feature-flagged.

---

## 1. Target Architecture (End State)

```
                                ┌────────────────────────────────────────┐
                                │            Client (Web / CLI)          │
                                └────────────────────┬───────────────────┘
                                                     │ HTTPS
                                                     ▼
                                ┌────────────────────────────────────────┐
                                │   nginx / traefik (TLS, rate limit)    │
                                └────────┬─────────────────────┬─────────┘
                                         │                     │
                              ┌──────────┴───────┐    ┌────────┴────────┐
                              │  backend pod #1  │ …  │ backend pod #N  │
                              │  gunicorn+uvicorn│    │ gunicorn+uvicorn│
                              └─────┬──┬──┬─┬───┘    └─────────────────┘
                                    │  │  │ │
                ┌───────────────────┘  │  │ └───────────────────────┐
                │                      │  │                         │
                ▼                      ▼  ▼                         ▼
        ┌──────────────┐       ┌────────────────┐         ┌──────────────────┐
        │   Postgres   │       │     Redis      │         │     Qdrant       │
        │   primary    │◄──────│  cache + queue │         │  dense + sparse  │
        │   + replica  │       │  (ARQ jobs)    │         │  (vectors)       │
        └──────────────┘       └────────┬───────┘         └──────────────────┘
                                        │
                                        ▼
                            ┌────────────────────────┐
                            │   ARQ worker pool      │
                            │   (ingestion jobs)     │
                            └──┬─────────────────┬───┘
                               │                 │
                               ▼                 ▼
                       ┌─────────────┐   ┌──────────────────┐
                       │   MinIO     │   │  LLM providers   │
                       │   (S3 API)  │   │ (Groq / Gemini)  │
                       └─────────────┘   └──────────────────┘
```

Key separations:
- **API pods**: stateless, handle HTTP; never block on heavy work
- **Worker pods**: pull from Redis queue, do ingestion / embedding / upserts
- **State**: Postgres (transactional), Qdrant (vectors), Redis (cache + queue), MinIO (blobs)

---

## 2. Tech Decisions

| Concern | Choice | Why |
|---|---|---|
| HTTP framework | FastAPI (kept) | Async, auto OpenAPI, Pydantic |
| ASGI server | gunicorn + uvicorn workers | `gunicorn -k uvicorn.workers.UvicornWorker -w 4` — multi-worker + graceful reload |
| Reverse proxy | nginx | TLS termination, rate limit, static health endpoints |
| Job queue | **ARQ** (Redis-backed, async-native) | Stays in Python+asyncio; lighter than Celery; reuses Redis you already run |
| Transactional DB | Postgres 16 (kept) | Mature, indexed reads, JSON support, easy replication |
| DB pooler | PgBouncer (transaction mode) | Multiplex 5000 clients → 50 server connections |
| DB ORM | SQLAlchemy 2.x async (kept) | Already in use |
| Migrations | Alembic (kept) | Run in CI on deploy |
| Cache | Redis (kept) | Already used for L2 RAG cache; reuse for queue + read-through cache |
| Vector DB | Qdrant (kept) | Already integrated; gRPC + sparse vectors |
| Object storage | **MinIO** (S3 API) | Free, S3-compatible, runs in docker-compose, swap to AWS S3/R2 with zero code change |
| Auth (Phase 2) | JWT (RS256) + API keys | Two flows, one `Principal` |
| Rate limiting | `slowapi` (Redis backend) | Per-user/per-endpoint, returns 429 + Retry-After |
| Tracing | OpenTelemetry → Jaeger | Trace query → retrieval → LLM chain |
| Metrics | Prometheus (kept) + Grafana | Already partially wired |
| Logs | loguru → JSON to stdout | 12-Factor; scraped by log shipper in prod |

---

## 3. Phased Execution Plan

Each phase: **scope → deliverables → files → acceptance criteria → estimated effort**.
Phases are sequenced so each is shippable on its own. Do not start phase N+1 until phase N's acceptance criteria pass.

### Phase 0 — Foundations (decision revised 2026-05-29: full auth removal)

**Scope:** Strip broken auth end-to-end. No flag, no stub, no dead code. Auth returns as a greenfield design in Phase 8.

**What was deleted:**
- `backend/auth/` (Principal, API-key hashing)
- `backend/routers/auth.py` (key issuance endpoint)
- `backend/models/auth.py`
- `backend/repos/api_keys.py`, `backend/repos/users.py`
- `backend/migrations/versions/001_initial.py` (users + api_keys tables)

**What changed:**
- `backend/deps.py` — only `get_pipeline` remains
- `backend/config.py` — removed `api_key_prefix`, `bootstrap_token`, `auth_disabled`
- `backend/routers/{query,ingest,collections}.py` — `Principal` dependency removed; `user_id` hardcoded to `"dev-user"` (placeholder until Phase 8)
- `backend/main.py` — auth router no longer mounted
- `backend/README.md` — endpoint table updated; quickstart no longer references bootstrap token

**What remained (dormant until later phases):**
- `backend/repos/base.py` — SQLAlchemy async engine + session factory (Phase 2+ will use)
- `backend/migrations/env.py` — Alembic config (Phase 2 adds first new migration)

**Acceptance (no formal tests written per scope decision):**
- `git grep -E 'backend\.auth|get_principal|Principal\b'` returns no source matches
- `backend/migrations/versions/` is empty
- `docker compose up` boots backend; `/v1/query` and `/v1/ingest` accept requests with no `Authorization` header

**Deferred from this phase (will land in Phase 1 with infra work):**
- pyproject.toml, ruff/mypy/pytest config
- pre-commit hooks
- Makefile
- `.env.example` template

---

### Phase 1 — Infrastructure Stack (2–3 days)

**Scope:** Get docker-compose to look like the target architecture.

**Deliverables:**
- `docker-compose.yml` gains: `minio`, `pgbouncer`, `nginx` services.
- Backend service runs `gunicorn -k uvicorn.workers.UvicornWorker -w 4`.
- New `worker` service runs `arq backend.workers.WorkerSettings`.
- nginx config: TLS termination (self-signed for dev), upstream to backend pods, request size limits, basic rate limit.
- MinIO console exposed on `:9001`; pre-create buckets `raw-uploads`, `exports`, `model-artifacts` via init container.
- PgBouncer in transaction-pooling mode; backend connects through PgBouncer, not directly.
- `docker-compose.gpu.yml` overlay updated for new services.
- Healthcheck stanzas on every service.

**Files touched:**
- [docker-compose.yml](docker-compose.yml) — add services
- [Dockerfile.backend](Dockerfile.backend) — switch entrypoint to gunicorn
- New: `Dockerfile.worker`, `infra/nginx.conf`, `infra/pgbouncer.ini`, `infra/minio-init.sh`

**Acceptance:**
- `docker compose up` brings up the full stack and all healthchecks pass
- `docker compose scale backend=3 worker=2` works (proves statelessness)
- `mc alias set local http://minio:9000 ... && mc ls local/` lists buckets
- nginx serves `/healthz` and proxies `/v1/*` to backend pool

---

### Phase 2 — Object Storage Layer (2 days)

**Scope:** Replace local-disk uploads with S3 (MinIO). Add deduplication.

**Deliverables:**
- New module `storage/` with:
  - `storage/object_store.py` — `S3ObjectStore` class wrapping `boto3` async (`aioboto3`)
  - `storage/keys.py` — key scheme: `{user_id}/{doc_id}/raw/{filename}` and `{user_id}/{doc_id}/processed/manifest.json`
  - `storage/factory.py` — pluggable backend (S3 / local for tests)
- Content-hash-based deduplication: SHA-256 of file contents → `documents` table has `content_hash` unique-per-user; re-upload of same content returns existing `doc_id`.
- Presigned URL support: for files >5MB, client gets presigned PUT URL and uploads directly to MinIO (offloads bandwidth from backend).
- Migration `002_documents_table.py`: `documents(id, user_id, content_hash, file_name, mime_type, size_bytes, s3_key, collection, status, created_at, deleted_at)`.

**Files touched:**
- [backend/routers/ingest.py](backend/routers/ingest.py) — switch from local file write to S3 upload (initially keep sync flow)
- New: `storage/`, `backend/repos/documents.py`, `backend/migrations/versions/002_documents.py`
- [backend/config.py](backend/config.py) — add `s3_endpoint`, `s3_access_key`, `s3_secret_key`, `s3_bucket_raw`

**Acceptance:**
- Upload a 30MB PDF → ends up at `minio/raw-uploads/{user_id}/{doc_id}/raw/file.pdf`
- Re-upload same file → returns existing `doc_id`, no new S3 object written
- Presigned URL flow works end-to-end via curl
- Local disk `data/uploads/` is no longer touched

---

### Phase 3 — Async Job Queue + Ingestion Refactor ✅ COMPLETED (2026-06-01)

**Scope:** Decouple HTTP request from ingestion work. This is the biggest scalability fix.

**What was delivered:**
- ARQ worker setup (`backend/workers/`):
  - `arq_settings.py` — `WorkerSettings` with Redis connection, max_jobs, job_timeout, on_startup/on_shutdown hooks
  - `queue_client.py` — `enqueue_ingest_job()` helper; returns arq job ID
  - `tasks.py` — `ingest_document_task(ctx, doc_id, ...)` — runs the full ingestion pipeline and updates document status
- `backend/services/ingestion_service.py` — orchestrates load → chunk → embed → upsert with per-chunk progress emission via `_ChunkProgressEmitter` and `on_batch_progress` callback
- `backend/services/redis_event_bus.py` — `RedisEventBus`: publishes SSE progress events to Redis pub/sub so all backend pods can fan-out to connected clients
- `backend/services/pipeline_factory.py` — lazy singleton factory; constructs and caches the pipeline instance shared across worker tasks
- `backend/services/orphan_sweeper.py` — `OrphanSweeper`: periodic background task that detects stuck jobs via `processing_started_at` lease timeout and resets them to `queued`
- DB migration `003`: adds `processing_started_at` column + index on `documents` table; `mark_ready_if_processing()` uses an atomic `WHERE status='processing'` guard
- Prometheus metrics: `ingest_jobs_queued_total`, `ingest_jobs_failed_total`, `ingest_jobs_inflight` (gauge)
- Worker runs as a separate Docker container (`Dockerfile.worker`); entry: `arq backend.workers.arq_settings.WorkerSettings`

**Files added:**
- `backend/workers/arq_settings.py`
- `backend/workers/queue_client.py`
- `backend/workers/tasks.py`
- `backend/services/ingestion_service.py`
- `backend/services/redis_event_bus.py`
- `backend/services/pipeline_factory.py`
- `backend/services/orphan_sweeper.py`

**Acceptance (verified):**
- POST a PDF → response in <200ms with `job_id`; worker picks it up and updates document status to `ready`
- Per-chunk SSE progress events flow from worker → Redis pub/sub → all connected backend pods
- Stuck-job detection: if `processing_started_at` exceeds lease timeout, `OrphanSweeper` resets the job to `queued`
- `ingest_jobs_inflight` gauge correctly tracks concurrent in-progress jobs
- Conditional `mark_ready_if_processing` prevents a race where a late-arriving duplicate marks an already-failed job as ready

---

### Phase 4 — Resource CRUD (Documents + Collections) (3 days)

**Scope:** Build the missing REST endpoints with proper pagination, filtering, soft-delete.

**Deliverables:**
- **Documents resource** (`backend/routers/documents.py`):
  - `POST /v1/documents` — already done in Phase 3
  - `GET /v1/documents?collection=X&status=ready&cursor=...&limit=50` — cursor pagination
  - `GET /v1/documents/{id}` — metadata + chunk count
  - `DELETE /v1/documents/{id}` — soft delete; background job purges vectors from Qdrant + S3 object
  - `GET /v1/documents/{id}/chunks?cursor=...&limit=20` — paginated chunks for debugging
- **Collections resource** (`backend/routers/collections.py` — flesh out existing):
  - `POST /v1/collections` — create collection, also creates Qdrant collection
  - `GET /v1/collections` — list with per-collection stats (doc count, total chunks)
  - `GET /v1/collections/{name}` — details
  - `PATCH /v1/collections/{name}` — update description
  - `DELETE /v1/collections/{name}` — guarded (must be empty or `?force=true`)
- Migration: `collections(name, description, owner_id, embed_model, created_at)`.
- Cursor pagination helper: `utils/pagination.py` — base64-encoded `{last_id, last_created_at}` cursors.

**Files touched:**
- [backend/routers/collections.py](backend/routers/collections.py) — expand
- New: `backend/routers/documents.py`, `backend/repos/collections.py`, `utils/pagination.py`
- New: `backend/migrations/versions/004_collections.py`
- `backend/workers/delete_document_job.py` — async vector + S3 cleanup

**Acceptance:**
- Full CRUD on documents and collections via curl
- Pagination: `?limit=10` returns 10 + `next_cursor`; follow cursor → next 10
- Delete document → vectors gone from Qdrant within 30s
- Delete non-empty collection → 409 Conflict (forces explicit `?force=true`)

---

### Phase 5 — Query Enhancements (3 days)

**Scope:** Streaming, search-only, idempotency.

**Deliverables:**
- `POST /v1/query/stream` — Server-Sent Events. Streams LLM tokens as `data: {token}\n\n`. Closes with `event: done\n` + final metadata (sources, timings).
- `POST /v1/search` — vector retrieval only, no LLM call. Returns ranked chunks. Cheap, useful for UI "instant search" and eval.
- **Idempotency keys**: `Idempotency-Key` header on POST endpoints. Redis stores `(user_id, idempotency_key) → response` for 24h. Replay returns cached response.
- Request/response logging middleware: structured JSON log per request (request_id, user_id, route, status, latency_ms, cache_hit).

**Files touched:**
- New: `backend/routers/search.py`
- [backend/routers/query.py](backend/routers/query.py) — add streaming endpoint
- New: `backend/middleware/idempotency.py`, `backend/middleware/access_log.py`
- [pipeline/rag_pipeline.py](pipeline/rag_pipeline.py) — expose `stream_query()` (yield tokens)
- [rag/base_rag.py](rag/base_rag.py) — add streaming generate path
- [llm/contracts/base_llm.py](llm/contracts/base_llm.py) — add `stream()` method

**Acceptance:**
- `curl -N` against `/v1/query/stream` shows tokens flowing in real-time
- Same `Idempotency-Key` replayed within 24h returns identical cached response (verifiable in logs)
- `/v1/search` returns chunks in <300ms (no LLM hop)

---

### Phase 6 — Conversations (2–3 days)

**Scope:** Make chat history a server-side resource.

**Deliverables:**
- Conversations + messages tables:
  - `conversations(id, user_id, title, collection, created_at, updated_at, deleted_at)`
  - `messages(id, conversation_id, role, content, sources, created_at)` — sources is JSON
- Endpoints:
  - `POST /v1/conversations` — create (auto-generate title from first message)
  - `GET /v1/conversations?cursor=...` — list
  - `GET /v1/conversations/{id}` — with last N messages (default 20)
  - `POST /v1/conversations/{id}/messages` — send user message; server fetches recent history, calls pipeline, persists user + assistant messages
  - `GET /v1/conversations/{id}/messages?cursor=...` — paginated message history
  - `DELETE /v1/conversations/{id}` — soft delete
- Client no longer passes `conversation_history` in `/v1/query` body for chat use cases.

**Files touched:**
- New: `backend/routers/conversations.py`, `backend/repos/conversations.py`
- New: `backend/migrations/versions/005_conversations.py`
- New: `backend/models/conversation.py`

**Acceptance:**
- Multi-turn chat works: create conversation, send 5 messages, history persists and is replayed correctly
- Pagination works on both conversations list and messages within a conversation
- Soft-deleted conversations don't appear in list but data is retained for 30 days

---

### Phase 7 — Cross-Cutting Concerns (3–4 days)

**Scope:** Rate limits, error responses, tracing, graceful shutdown.

**Deliverables:**
- **Rate limiting** (`slowapi` + Redis):
  - Global: 1000 req/min per user (default)
  - `/v1/query`: 60 req/min per user
  - `/v1/documents` (POST): 30 req/hour per user
  - Returns 429 + `Retry-After` + `X-RateLimit-Remaining`
- **RFC 7807 error responses**:
  - `backend/errors/problem.py` — `Problem` Pydantic model
  - Global exception handler converts `HTTPException` and custom exceptions to Problem JSON
  - Stable `error_code` strings (e.g., `DOCUMENT_NOT_FOUND`, `RATE_LIMITED`, `INGESTION_TIMEOUT`)
- **OpenTelemetry tracing**:
  - Auto-instrument FastAPI, SQLAlchemy, redis, httpx
  - Manual spans around: `pipeline.query`, `retriever.retrieve`, `llm.generate`, `cache.lookup`
  - Export to Jaeger (added to docker-compose for dev)
- **Graceful shutdown**:
  - SIGTERM handler → set `app.state.ready = False` → drain in-flight requests up to 30s → tear down pipeline
  - nginx upstream healthcheck reads `/readyz`; failing readiness removes pod from pool
- **`/readyz`** with parallel dep checks (1s timeout each): Postgres ping, Redis ping, Qdrant ping. Returns 503 if any fail.

**Files touched:**
- New: `backend/middleware/rate_limit.py`, `backend/errors/problem.py`, `backend/errors/handlers.py`
- New: `backend/observability/tracing.py`
- [backend/routers/health.py](backend/routers/health.py) — split into `/healthz` (liveness) and `/readyz` (readiness)
- [backend/main.py](backend/main.py) — wire shutdown handler, error handlers, tracing init

**Acceptance:**
- Hammer `/v1/query` past the rate limit → 429 with proper headers
- Trigger a `DOCUMENT_NOT_FOUND` → response is RFC 7807 JSON with `error_code`
- Jaeger UI shows trace spanning HTTP → pipeline → retriever → LLM
- `docker stop backend-1` (with in-flight requests) → all complete cleanly, no 500s
- `/readyz` returns 503 when Postgres is paused, recovers automatically

---

### Phase 8 — Auth Reintroduction (3–4 days)

**Scope:** Greenfield rebuild — Phase 0 deleted the old code entirely, so there is no legacy auth to integrate with. Design fresh.

**Deliverables:**
- **Two auth flows, one `Principal`**:
  - JWT (RS256) for end users — `/v1/auth/login` issues access (15min) + refresh (7d). Refresh rotates.
  - API keys for programmatic — existing flow polished
- **Scopes/roles** on Principal: `documents:read`, `documents:write`, `collections:admin`, `admin:*`
- Endpoint dependencies declare required scopes: `principal: Principal = Depends(require_scope("documents:write"))`
- Endpoints:
  - `POST /v1/auth/register` — email + password (Argon2 hash)
  - `POST /v1/auth/login` — returns access + refresh
  - `POST /v1/auth/refresh`
  - `POST /v1/auth/logout` — revokes refresh token (Redis blacklist)
  - `GET /v1/auth/me`
  - `POST /v1/auth/api-keys` — create scoped API key
  - `GET /v1/auth/api-keys` — list (prefix only)
  - `DELETE /v1/auth/api-keys/{id}`
- Rate limits keyed on `principal.user_id`, not IP.
- No legacy flag to flip — auth was deleted in Phase 0; this phase builds it from scratch.

**Files touched:**
- New: `backend/auth/jwt_service.py`, `backend/auth/password.py`, `backend/auth/scopes.py`
- [backend/auth/principal.py](backend/auth/principal.py) — add scopes
- [backend/deps.py](backend/deps.py) — `get_principal` resolves either JWT or API key
- [backend/routers/auth.py](backend/routers/auth.py) — full rewrite
- New: `backend/migrations/versions/006_refresh_tokens.py`

**Acceptance:**
- Register → login → call `/v1/query` with bearer JWT → works
- Same call with API key → also works
- Endpoint requiring `documents:write` rejects a key with only `documents:read` (403)
- Expired access token → 401 with `WWW-Authenticate: Bearer error="invalid_token"`
- Refresh rotation: old refresh becomes invalid after use

---

### Phase 9 — Observability & SLOs (2–3 days)

**Scope:** You can't operate what you can't see.

**Deliverables:**
- Grafana dashboards (provisioned as code in `infra/grafana/`):
  - **API panel**: QPS, p50/p95/p99 latency per route, error rate by `error_code`
  - **RAG panel**: cache hit rate (L1/L2/semantic), retrieval latency, LLM latency, decompose-vs-simple ratio
  - **LLM cost panel**: tokens per minute per model, estimated $/hour
  - **Infra panel**: Postgres connections, Redis ops/s, Qdrant query latency, MinIO storage used
  - **Jobs panel**: queue depth, job duration p95, failure rate, DLQ size
- **SLOs** defined in `SLO.md`:
  - 99% of `/v1/query` < 3s
  - 99.9% availability over 30 days
  - 95% of ingestion jobs succeed on first attempt
- **Alerts** (Prometheus Alertmanager rules):
  - SLO burn rate (1h and 6h windows)
  - Queue depth > 1000 for 5min
  - Any service /readyz failing for 2min
  - LLM error rate > 5% for 5min

**Files touched:**
- New: `infra/grafana/dashboards/*.json`, `infra/prometheus/alerts.yml`, `SLO.md`
- New: `backend/observability/llm_cost.py` — actually compute cost from token counts (currently TODO)

**Acceptance:**
- Open Grafana → all panels populated
- Synthetically push p95 latency over budget → alert fires
- Force a DLQ entry → alert fires

---

### Phase 10 — Load Test & Hardening (3–5 days)

**Scope:** Prove it holds 2K concurrent users.

**Deliverables:**
- **k6 load test scripts** in `tests/load/`:
  - `query_load.js` — ramp 0 → 2000 VUs over 5min, hold 10min, ramp down
  - `ingest_load.js` — 200 concurrent uploads of 5MB files
  - `mixed_workload.js` — realistic ratio: 90% query, 5% ingest, 5% list
- Tune based on results:
  - gunicorn worker count
  - Postgres `max_connections`, PgBouncer pool size
  - Redis `maxmemory` and eviction policy
  - Qdrant gRPC pool settings
- **Runbook** (`RUNBOOK.md`):
  - Deployment procedure
  - Incident response (DB down, Qdrant slow, queue backed up)
  - Common diagnostic queries
  - Capacity planning model
- **Canary deploy pattern** documented (5% traffic → watch → promote).

**Acceptance:**
- 2000 VUs sustained → p99 query latency stays under SLO
- No 5xx errors during load test
- Memory and connections stable (no leaks)
- Recovery from a simulated Postgres failover under 30s

---

## 4. Files & Directory Layout (End State)

```
Scalable RAG/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── deps.py
│   ├── auth/                       # JWT, API key, password, scopes
│   ├── errors/                     # Problem details, handlers
│   ├── middleware/                 # request_id, rate_limit, idempotency, access_log
│   ├── models/                     # Pydantic request/response models
│   ├── observability/              # metrics, tracing, llm_cost
│   ├── repos/                      # users, api_keys, documents, jobs, collections, conversations
│   ├── routers/                    # health, auth, query, search, documents, jobs, collections, conversations
│   ├── workers/                    # ARQ jobs: ingest, delete_document, reindex
│   └── migrations/                 # Alembic
├── storage/                        # S3/MinIO client
├── pipeline/, rag/, agents/, llm/, cache/, vectorstore/, chunking/, optimizer/, utils/  # unchanged in spirit, refactored as needed
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/                        # current test_*.py moved here
│   └── load/                       # k6 scripts
├── infra/
│   ├── nginx.conf
│   ├── pgbouncer.ini
│   ├── grafana/
│   └── prometheus/
├── docker-compose.yml              # full stack
├── docker-compose.gpu.yml          # overlay
├── Dockerfile.backend
├── Dockerfile.worker               # new
├── pyproject.toml                  # consolidated config
├── Makefile
├── .env.example                    # secrets-free template
├── PLAN.md                         # this file
├── SLO.md                          # phase 9
├── RUNBOOK.md                      # phase 10
├── WORKFLOW.md                     # existing
└── Optimizations.md                # existing
```

---

## 5. CAP Theorem — Practical Application

You asked for "C and A". Honest answer: in any distributed system you must tolerate partitions. So the real choice is **per-component**:

| Component | Role | CAP class | Reasoning |
|---|---|---|---|
| Postgres (primary + sync replica) | Users, API keys, jobs, documents metadata | **CP** | Strong consistency; replica acks before commit. Brief unavailability during failover acceptable. |
| Qdrant | Vectors | **AP** (eventually consistent) | Slightly stale chunk visibility (seconds) is fine for RAG; availability matters more. |
| Redis (master + replica) | Cache + queue | **AP** | Cache miss on partition is acceptable; queue uses Redis Streams for at-least-once delivery. |
| MinIO | Raw files | **AP** | Write durability + read availability. Eventual consistency on metadata. |

**For a user request, end-to-end the system behaves as CA** (single-region, partitions rare). When you scale to multi-region (50K+ users), you'll revisit this and likely accept eventual consistency for replicated metadata.

---

## 6. API Conventions (apply to every endpoint)

- **Versioning**: all routes under `/v{N}/`. Never break v1.
- **Resource naming**: plural nouns (`/documents`, `/collections`). Verbs only in actions (`/v1/documents/{id}/reindex`).
- **HTTP semantics**:
  - `200 OK` — success with body
  - `201 Created` — resource created synchronously (`Location` header)
  - `202 Accepted` — async work queued (`Location` → job URL)
  - `204 No Content` — success, no body (deletes)
  - `400 Bad Request` — validation error
  - `401 Unauthorized` — missing/invalid auth
  - `403 Forbidden` — auth OK but lacks scope
  - `404 Not Found`
  - `409 Conflict` — state conflict (delete non-empty collection)
  - `413 Payload Too Large`
  - `422 Unprocessable Entity` — Pydantic validation (FastAPI default)
  - `429 Too Many Requests` — rate limited
  - `5xx` — server fault
- **Error body**: RFC 7807 Problem Details
  ```json
  {
    "type": "https://api.example.com/errors/document-not-found",
    "title": "Document not found",
    "status": 404,
    "error_code": "DOCUMENT_NOT_FOUND",
    "detail": "No document with id 'abc123' for this user",
    "instance": "/v1/documents/abc123",
    "request_id": "01HNX..."
  }
  ```
- **Pagination**: cursor-based
  ```
  GET /v1/documents?limit=50&cursor=eyJpZCI6...
  → { "items": [...], "next_cursor": "eyJpZCI6...", "has_more": true }
  ```
- **Filtering**: explicit query params, never DSL. `?collection=foo&status=ready`.
- **Idempotency**: `Idempotency-Key: <uuid>` header on every POST.
- **Headers always returned**: `X-Request-ID`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`.
- **No PII in URLs or logs.** User IDs OK; emails / file contents never.

---

## 7. Risks & Open Questions

**Risks:**
1. **Refactor scope is large** — phases 1–7 are ~3 weeks of focused work. Don't try to parallelize phases.
2. **Existing tests are e2e only** — refactoring will break them. Need to add unit/integration coverage in Phase 0 to avoid flying blind.
3. **Auth re-add (Phase 8) touches every endpoint** — must be flag-gated so we can ship incrementally.
4. **MinIO disk usage** — set lifecycle policy to expire old uploads, or disk fills up in dev.
5. **Qdrant Cloud free tier** — confirm it can hold the load; may need to migrate to self-hosted Qdrant in docker-compose.

**Open questions to resolve before Phase 1:**
1. **Deployment target?** Local Docker only, or also Kubernetes / cloud? Affects nginx vs ingress, secrets mgmt, etc.
2. **Domain & TLS?** Real domain → real cert (Let's Encrypt). Local-only → self-signed is fine.
3. **Conversation history retention?** 30 days soft-delete? 90? Forever?
4. **Multi-tenancy model?** Single org per user, or users belong to organizations with shared collections? Affects auth schema in Phase 8.
5. **PII / compliance?** Are documents potentially sensitive (PHI, PCI)? Affects logging, encryption-at-rest, audit trails.
6. **Streaming over SSE or WebSocket?** SSE is simpler; WebSocket allows client→server mid-stream messages (cancel, modify). I've assumed SSE.
7. **Will users see other users' collections?** Or are all collections private? Affects collection list endpoint authorization.

---

## 8. Anti-Goals (explicitly out of scope)

- **Multi-region active-active** — overkill at 2K users
- **Custom auth (OAuth provider)** — use JWT for now; integrate with external IdP (Auth0, Keycloak) only if needed
- **GraphQL** — REST is sufficient and aligned with your learning goals
- **Microservices split** — keep monolith + workers; split only if a service has different scaling needs
- **Kubernetes** in Phase 0–9 — docker-compose is fine for 2K users on a single beefy box / VM. K8s can come later.
- **Custom load balancer / service mesh** — nginx is enough
- **Real-time collaboration** features — not part of RAG product

---

## 9. Recommended Execution Order

**Week 1:** Phase 0 (foundations) + Phase 1 (infra stack)
**Week 2:** Phase 2 (storage) + Phase 3 (async ingestion) — this is where the system becomes truly scalable
**Week 3:** Phase 4 (CRUD) + Phase 5 (query enhancements)
**Week 4:** Phase 6 (conversations) + Phase 7 (cross-cutting)
**Week 5:** Phase 8 (auth)
**Week 6:** Phase 9 (observability) + Phase 10 (load test + harden)

Total: ~6 weeks of focused work. Compressible if you skip phases you don't need (e.g., conversations if you don't need chat UX).

---

## 10. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-29 | Use ARQ (not Celery) | Async-native, fewer moving parts, reuses Redis you run |
| 2026-05-29 | MinIO over Cloudflare R2 | $0 + offline dev + same S3 API |
| 2026-05-29 | Keep Postgres (not switch to NoSQL) | Read-heavy is solved by Qdrant + Redis cache, not by switching DBs |
| 2026-05-29 | **Delete auth entirely (revised from "disable via flag")** | User direction: keep the codebase clean rather than carry half-broken auth + a bypass flag. Phase 8 rebuilds from scratch with no legacy to integrate. |
| 2026-05-29 | gunicorn + uvicorn workers | Multi-worker for CPU parallelism + graceful reload |
| 2026-05-29 | SSE for streaming (not WebSocket) | Simpler; can revisit if cancel/modify needed |

---

## 11. What I Need From You Before We Start

Answer the 7 open questions in section 7, plus confirm:
1. Are you OK with the 6-week timeline? Want to compress / drop phases?
2. Do you have a deployment target in mind (your laptop, a VPS, AWS, etc.)?
3. Any phases you want to reorder or skip?
4. Any tech choices in section 2 you want to challenge?

Once these are settled, we start Phase 0.
