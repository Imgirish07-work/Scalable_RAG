# Scalable RAG — Backend

FastAPI layer wrapping the RAG pipeline. API-key auth, Postgres-backed users.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET  | `/healthz` | none | Liveness |
| GET  | `/readyz`  | none | Readiness (waits for pipeline.initialize()) |
| GET  | `/metrics` | none | Prometheus exposition |
| POST | `/v1/auth/keys` | bootstrap token | Issue an API key |
| POST | `/v1/query` | API key | Synchronous RAG query |
| POST | `/v1/ingest` | API key | Multipart upload + ingest |
| GET  | `/v1/collections` | API key | List configured collections |

## Quickstart

```powershell
Copy-Item .env.example .env
# Set GROQ_API_KEY or GEMINI_API_KEY, and BACKEND_BOOTSTRAP_TOKEN.

docker compose --profile dev up --build
docker compose exec backend alembic -c backend/migrations/alembic.ini upgrade head

# Issue a key
curl -X POST http://localhost:8000/v1/auth/keys `
  -H "X-Bootstrap-Token: $env:BACKEND_BOOTSTRAP_TOKEN" `
  -H "Content-Type: application/json" `
  -d '{\"email\":\"me@example.com\",\"name\":\"dev\"}'

# Ingest
curl -X POST http://localhost:8000/v1/ingest `
  -H "Authorization: Bearer rag_YOUR_KEY" `
  -F "file=@./data/sample_docs/your-file.pdf" `
  -F "collection=my-docs"

# Query
curl -X POST http://localhost:8000/v1/query `
  -H "Authorization: Bearer rag_YOUR_KEY" `
  -H "Content-Type: application/json" `
  -d '{\"query\":\"summarize this\",\"collection\":\"my-docs\",\"top_k\":5}'
```

## Layout

```
backend/
├── main.py            FastAPI app + lifespan
├── config.py          BackendSettings
├── deps.py            get_pipeline, get_db, get_principal
├── auth/              Principal, API-key hashing
├── middleware/        request_id + access log
├── routers/           health, auth, query, ingest, collections
├── repos/             Async SQLAlchemy: users, api_keys
├── models/            Pydantic API shapes
├── observability/     Prometheus metrics
└── migrations/        Alembic
```

## Operational notes

- First boot: ~30-90 s (pip install + pipeline warmup). Watch for `Backend ready in N ms`.
- `/readyz` returns 503 until warmup completes.
- API keys are shown once on issuance; only the SHA-256 hash is stored.
- Set `RERANKER_ENABLED=false` if cross-encoder model files aren't available.
