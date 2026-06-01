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

### 1b. Async Ingestion Flow (Phase 3 — current production path)

```
HTTP POST /v1/ingest  (multipart upload)
  │
  ▼
backend/api/v1/documents.py
  │  writes document row (status=queued) to Postgres
  │  calls queue_client.enqueue_ingest_job(doc_id, ...)
  │  returns 202 Accepted + { job_id, document_id }
  │
  ▼ (Redis queue — arq)
backend/workers/tasks.py  ingest_document_task(ctx, doc_id, ...)
  │  sets document status=processing, records processing_started_at
  │
  ▼
backend/services/ingestion_service.py  IngestionService.run()
  │
  ├─ DocumentCleaner    chunking/document_cleaner.py
  │    normalizes raw text
  ├─ StructurePreserver chunking/structure_preserver.py
  │    preserves headings / tables / lists
  ├─ Chunker            chunking/chunker.py
  │    token-bounded chunks with overlap
  │
  └─ _ChunkProgressEmitter (on_batch_progress callback)
       for each batch of embedded chunks:
         QdrantStore.upsert()    vectorstore/qdrant_store.py
         RedisEventBus.publish() backend/services/redis_event_bus.py
           │  emits { doc_id, chunks_done, chunks_total } on Redis pub/sub
           ▼
         All backend pods subscribed → SSE push to connected clients
  │
  ▼
mark_ready_if_processing()   atomic WHERE status='processing' guard
  sets document status=ready in Postgres
```

**Stuck-job recovery:**
```
OrphanSweeper (periodic background task)   backend/services/orphan_sweeper.py
  │  SELECT documents WHERE status='processing'
  │    AND processing_started_at < NOW() - lease_timeout
  └─ resets orphaned rows to status=queued → re-enqueued on next worker poll
```

**Prometheus metrics emitted during async ingestion:**
- `ingest_jobs_queued_total` — incremented when a job is enqueued
- `ingest_jobs_inflight` (gauge) — tracks concurrent in-progress tasks
- `ingest_jobs_failed_total` — incremented on task exception

**Key files (Phase 3):**

| File | Role |
|------|------|
| `backend/workers/arq_settings.py` | ARQ `WorkerSettings`, Redis pool config |
| `backend/workers/queue_client.py` | `enqueue_ingest_job()` helper |
| `backend/workers/tasks.py` | `ingest_document_task` arq task |
| `backend/services/ingestion_service.py` | Orchestrates chunk → embed → upsert |
| `backend/services/redis_event_bus.py` | Redis pub/sub SSE fan-out |
| `backend/services/pipeline_factory.py` | Lazy pipeline singleton for workers |
| `backend/services/orphan_sweeper.py` | Lease-based stuck-job detector |

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
| `backend/workers/tasks.py` | ARQ task entry point for async ingestion    |
| `backend/services/ingestion_service.py` | Core ingestion orchestration with progress emission |
| `backend/services/redis_event_bus.py` | Redis pub/sub fan-out for SSE progress |
| `backend/services/orphan_sweeper.py` | Stuck-job lease detection and recovery |

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
