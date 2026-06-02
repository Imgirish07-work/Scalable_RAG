# RAG Pipeline — End-to-End Workflow

Quick-reference for the full pipeline architecture. Two entry points:
**Ingestion** (load documents) and **Query** (retrieve + generate).

---

## 1. Ingestion Flow

### 1a. Synchronous (legacy / direct pipeline call)

```
PDF file
  │
  ▼
DocumentCleaner          chunking/document_cleaner.py
  │  normalizes raw text (whitespace, encoding, artifacts)
  ▼
StructurePreserver       chunking/structure_preserver.py
  │  keeps headings, tables, lists intact before splitting
  ▼
Chunker                  chunking/chunker.py
  │  splits into token-bounded chunks with overlap
  ▼
QdrantStore.upsert()     vectorstore/qdrant_store.py
     embeds each chunk via BGE (vectorstore/embeddings.py)
     stores vector + payload in Qdrant collection
```

**Entry point:** `RAGPipeline.ingest(file_path, collection)`

---

### 1b. Async Ingestion Flow (current production path)

```
 BROWSER (React)
 ┌─────────────────────────────────────────────────────────┐
 │  1. POST /v1/ingest          → reserve doc_id           │
 │  2. PUT  <presigned-url>     → upload file to MinIO     │
 │  3. POST /v1/documents/{id}/finalize → trigger worker   │
 │  4. GET  /v1/documents/{id}/events   → open SSE stream  │
 └───────┬──────────────────────────────────────┬──────────┘
         │  HTTP/JSON                            │  SSE (text/event-stream)
         ▼                                       ▼
 ┌───────────────────────────────────────────────────────────────────────────┐
 │  FastAPI  (backend pod)                                                   │
 │                                                                           │
 │  [1] POST /v1/ingest                                                      │
 │      DocumentService.create_upload_session()                              │
 │        • INSERT document row  (status=pending)  → Postgres                │
 │        • generate presigned PUT URL             → MinIO                   │
 │        • return {doc_id, upload_url}                                      │
 │                                                                           │
 │  [3] POST /v1/documents/{doc_id}/finalize                                 │
 │      DocumentService.finalize()                                           │
 │        • validate doc row exists & is pending                             │
 │        • arq_pool.enqueue_job("ingest_document", doc_id, user_id)        │
 │        • return {job_id}                                                  │
 │                                                                           │
 │  [4] GET /v1/documents/{doc_id}/events  (SSE)                             │
 │      DocumentService.subscribe_to_events()                                │
 │        • RedisEventBus.subscribe(doc_id)   ←── Redis SUBSCRIBE            │
 │        • stream events as  text/event-stream until terminal phase         │
 └───────────────────┬────────────────────────────┬─────────────────────────┘
                     │                            │
          ┌──────────▼──────────┐     ┌───────────▼──────────────┐
          │      Redis          │     │       Postgres            │
          │                     │     │                           │
          │  • Arq job queue    │     │  documents table          │
          │  • Pub/sub channels │     │  status: pending          │
          │    events:{doc_id}  │     │         processing        │
          │  • Keepalive ticks  │     │         ready / failed    │
          └──────────┬──────────┘     │         duplicate / dlq   │
                     │                └───────────────────────────┘
          ┌──────────▼───────────────────────────────────────────────────────┐
          │  Arq Worker  (worker pod)                                        │
          │                                                                  │
          │  WorkerSettings                                                  │
          │    functions  = [ingest_document]                                │
          │    cron_jobs  = [_gpu_keepalive  every 3 min]                   │
          │    max_jobs   = 1   (GPU safety — serial execution)              │
          │                                                                  │
          │  on_startup:                                                     │
          │    • build_ingest_pipeline()  → RAGPipeline (ingest mode)       │
          │    • _run_warmup()  → embeds warmup_doc.txt chunks               │
          │    • IngestionService(store, pipeline, event_bus)                │
          │                                                                  │
          │  ── TASK: ingest_document(ctx, doc_id, user_id) ──               │
          │                                                                  │
          │  IngestionService.run()                                          │
          │    │                                                             │
          │    ├─ repo.update_status(doc_id, "processing")  → Postgres      │
          │    ├─ publish("processing")  ──────────────────────────────► Redis
          │    │                                                             │
          │    ├─ store.head_object(s3_key)  → MinIO                        │
          │    ├─ publish("downloading")  ─────────────────────────────► Redis
          │    │                                                             │
          │    ├─ _download_and_hash()                                      │
          │    │    • stream GET from MinIO                                  │
          │    │    • write to temp file + SHA-256 hash                      │
          │    ├─ publish("hashed")  ───────────────────────────────────► Redis
          │    │                                                             │
          │    ├─ _verify_mime_from_disk()  (python-magic)                  │
          │    │                                                             │
          │    ├─ dedup check: find_active_by_content_hash()  → Postgres    │
          │    │    if duplicate ──► cascade_delete → Qdrant + MinIO        │
          │    │                ──► hard_delete     → Postgres               │
          │    │                ──► publish("duplicate")  ──────────────► Redis
          │    │                ──► return                                   │
          │    │                                                             │
          │    ├─ repo.set_content_hash(doc_id, hash)  → Postgres           │
          │    ├─ publish("chunking")  ─────────────────────────────────► Redis
          │    │                                                             │
          │    └─ pipeline.ingest(temp_path, collection, user_id, doc_id    │
          │         on_batch_progress=ChunkProgressEmitter.emit)            │
          │                                                                  │
          │         ┌─────────────────────────────────────────┐             │
          │         │  RAGPipeline.ingest()  (ingest mode)    │             │
          │         │                                         │             │
          │         │  DocumentCleaner → StructurePreserver   │             │
          │         │  Chunker (800-char splits)              │             │
          │         │                                         │             │
          │         │  for each batch of chunks:              │             │
          │         │    dense_embed()  ← ONNX/GPU            │             │
          │         │    sparse_embed() ← SPLADE/GPU          │             │
          │         │    QdrantStore.upsert_batch()  ─────────┼──────────► Qdrant
          │         │    on_batch_progress(n, total)          │             │
          │         │      └─► publish("embedding", ...)  ────┼──────────► Redis
          │         │                                         │             │
          │         │  returns IngestionResult(chunks_stored) │             │
          │         └─────────────────────────────────────────┘             │
          │                                                                  │
          │    ├─ repo.mark_ready_if_processing()  → Postgres               │
          │    ├─ publish("ready")  ───────────────────────────────────► Redis
          │    └─ update Prometheus metrics                                  │
          │                                                                  │
          │  ── CRON: _gpu_keepalive()  (every 3 min) ──                    │
          │    • embeds 20 chunks from warmup_doc.txt                        │
          │    • keeps CUDA kernels hot, prevents 40s cold-start             │
          └──────────────────────────────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐     ┌──────────────────────┐
          │      MinIO          │     │      Qdrant           │
          │  (object store)     │     │  (vector store)       │
          │                     │     │                       │
          │  • presigned PUT    │     │  • dense vectors      │
          │  • GET stream       │     │  • sparse vectors     │
          │  • delete on dedup  │     │  • BM42 payloads      │
          └─────────────────────┘     │  • delete_by_doc_id   │
                                      └──────────────────────┘
```

**Redis pub/sub phase sequence** (terminal phase ends SSE stream):
```
processing → downloading → hashed → chunking →
embedding (×N batches, chunks_processed / chunks_total) →
ready  ✓  (or: duplicate / failed)
```

**Stuck-job recovery:**
```
OrphanSweeper (periodic background task)   backend/services/orphan_sweeper.py
  │  SELECT documents WHERE status='processing'
  │    AND processing_started_at < NOW() - lease_timeout
  └─ resets orphaned rows to status=queued → re-enqueued on next worker poll
```

**Prometheus metrics emitted during async ingestion:**
- `ingest_total{outcome}` — counter per terminal outcome (ready / failed / duplicate)
- `ingest_chunks_total` — total chunks embedded and stored
- `ingest_duration_seconds{outcome}` — histogram of end-to-end ingest time
- `ingest_jobs_inflight` (gauge) — concurrent in-progress tasks
- `ingest_jobs_failed_total{reason}` — incremented on task exception

**Key design decisions:**

| Decision | Why |
|----------|-----|
| `max_jobs=1` | GPU safety — serial ingestion prevents OOM from concurrent ONNX sessions |
| Presigned PUT URL | Client uploads directly to MinIO; API pod never buffers the file |
| Redis pub/sub for events | Worker and API are separate processes; in-process queue won't cross the boundary |
| `mark_ready_if_processing` guard | Sweeper may move the row to `failed` while the job is finishing; avoids status race |
| Keepalive cron every 3 min | GPU VRAM persists across jobs but CUDA kernel state decays after ~5-8 min idle |
| Ingest-mode pipeline | Worker skips LLM / agents / cache — only store + embeddings loaded |

**Key files:**

| File | Role |
|------|------|
| `backend/workers/arq_settings.py` | `WorkerSettings`, Redis config, GPU keepalive cron |
| `backend/workers/tasks.py` | `ingest_document` arq task entry point |
| `backend/services/ingestion_service.py` | Full ingest orchestration: download → hash → dedup → chunk → embed → upsert |
| `backend/services/redis_event_bus.py` | Redis pub/sub SSE fan-out |
| `backend/services/pipeline_factory.py` | Builds ingest-mode `RAGPipeline` for worker pod |
| `backend/services/orphan_sweeper.py` | Lease-based stuck-job detector |
| `pipeline/warmup_doc.txt` | Real-text warmup corpus (~15 chunks) used by startup warmup and keepalive cron |

---

## 2. Query Flow — Overview

```
PipelineQuery (query, collection, top_k)
  │
  ▼
RAGPipeline.query()      pipeline/rag_pipeline.py
  │
  ├─ CacheManager.get()  cache/cache_manager.py
  │   hit ──────────────────────────────────────► RAGResponse (cache_hit=True)
  │   miss ▼
  │
  ├─ should_decompose(query)?   agents/planner/complexity_detector.py
  │
  ├── NO  ──► SIMPLE PATH
  └── YES ──► AGENT PATH
```

---

## 3. Simple Path (direct factual / single-hop)

```
RAGPipeline
  │
  ▼
RAGFactory.create("simple")     rag/rag_factory.py
  │  injects DenseRetriever or HybridRetriever
  ▼
SimpleRAG.query()               rag/variants/simple_rag.py
  │
  ├─ pre_process()              rag/base_rag.py
  │    refines query using conversation history (1 LLM call if history present)
  │
  ├─ retrieve()                 rag/variants/simple_rag.py
  │    delegates to retriever ──► DenseRetriever   rag/retrieval/dense_retriever.py
  │                           └─► HybridRetriever  rag/retrieval/hybrid_retriever.py
  │                                │
  │                                ▼
  │                           QdrantStore.similarity_search_with_vectors()
  │                           vectorstore/qdrant_store.py
  │
  ├─ rank()                    rag/base_rag.py
  │    MMR re-ranking          rag/context/context_ranker.py
  │
  ├─ assemble_context()        rag/base_rag.py
  │    token-bounded context   rag/context/context_assembler.py
  │
  ├─ generate()                rag/base_rag.py
  │    grounded LLM call       llm/providers/  (Gemini / Groq pool)
  │
  └─ cache()                   rag/base_rag.py
       writes result to cache

  ▼
RAGResponse (rag_variant="simple")
```

**LLM calls:** 1 generation (+ 1 optional query-refinement if history present)

---

## 4. Agent Path (complex / multi-aspect queries)

```
RAGPipeline
  │
  ▼
AgentOrchestrator.run()         agents/agent_orchestrator.py
  │
  ├─ QueryPlanner.decompose()   agents/planner/query_planner.py
  │    strong LLM breaks query into ≤ 3 focused sub-queries
  │    reads COLLECTIONS dict to write targeted sub-queries
  │
  ├─ [parallel] for each sub-query:
  │    │
  │    ├─ ChunkRetriever.retrieve()   agents/retriever/
  │    │    calls SimpleRAG.retrieve() on the target collection
  │    │    returns raw chunks (no generation)
  │    │
  │    └─ ChunkQualityGate.classify() agents/quality/chunk_quality_gate.py
  │         deterministic strong / weak / failed classification
  │
  ├─ ContextFusion.fuse()       agents/fusion/context_fusion.py
  │    slot reservation + MMR + token budget across all sub-query results
  │
  └─ Synthesizer (LLM)
       strong LLM generates final answer from fused context

  ▼
RAGResponse (rag_variant="agent")
```

**LLM calls:** 1 decompose + 1 synthesis = 2 calls minimum
(fast LLM used for rewriting if `GROQ_MODEL_FAST` is set)

---

## 5. Cache Layer

```
CacheManager               cache/cache_manager.py
  │
  ├─ L1: In-memory (exact match, TTL)    cache/strategies/
  ├─ L2: Redis (semantic similarity)     cache/backend/
  └─ L3: Qdrant semantic cache           cache/strategies/
```

Checked **before** routing (Simple or Agent). Written after any cache miss.

---

## 6. LLM Layer

```
LLMFactory                 llm/llm_factory.py
  │
  ├─ create_groq_pool()    llm/providers/groq_model_pool.py
  │    round-robin across multiple Groq API keys
  │
  └─ create_rate_limited() llm/providers/
       Gemini fallback with rate-limit handling

RateLimiter                llm/rate_limiter/
ProviderHealth             llm/provider_health.py
```

---

## 7. Key Configuration Points

| What                     | Where                          |
|--------------------------|--------------------------------|
| Search mode (dense/hybrid) | `SEARCH_MODE` in test file or `QdrantStore(search_mode=)` |
| Qdrant collection        | `COLLECTION` constant — must match across ingest + configure_agents + PipelineQuery |
| Collection descriptions  | `COLLECTIONS` dict — read by QueryPlanner to route sub-queries |
| Max sub-queries          | `_MAX_SUB_QUERIES = 3` in `agents/planner/query_planner.py` |
| Fast LLM for agents      | `GROQ_MODEL_FAST` in `.env` / `config/settings.py` |
| Cache settings           | `config/settings.py` (TTL, Redis URL, thresholds) |
| Complexity threshold     | `agents/planner/complexity_detector.py` |

---

## 8. Entry-Point Files

| File                      | Purpose                                      |
|---------------------------|----------------------------------------------|
| `test_real_pipeline.py`   | Full end-to-end test (ingest + query)        |
| `pipeline/rag_pipeline.py`| Main orchestrator — the only class with the agent layer |
| `pipeline/models/pipeline_request.py` | `PipelineQuery` — public query model |
| `rag/rag_factory.py`      | Creates `SimpleRAG` with correct retriever   |
| `agents/agent_orchestrator.py` | Runs the full agent path               |
| `vectorstore/qdrant_store.py` | All vector DB operations              |
| `backend/workers/arq_settings.py` | `WorkerSettings`, Redis config, GPU keepalive cron |
| `backend/workers/tasks.py` | `ingest_document` arq task entry point |
| `backend/services/ingestion_service.py` | Full ingest orchestration: download → hash → dedup → chunk → embed → upsert |
| `backend/services/redis_event_bus.py` | Redis pub/sub fan-out for SSE progress |
| `backend/services/orphan_sweeper.py` | Stuck-job lease detection and recovery |
| `pipeline/warmup_doc.txt` | Real-text warmup corpus for startup warmup and GPU keepalive cron |

---

## 9. Routing Summary

```
Query arrives
  │
  ├─ Cache hit?          ──► return cached RAGResponse
  │
  ├─ should_decompose()? ──► YES: AgentOrchestrator (2+ LLM calls)
  │
  └─ default             ──► SimpleRAG (1 LLM call)
```
