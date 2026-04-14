# RAG Pipeline — End-to-End Workflow

Quick-reference for the full pipeline architecture. Two entry points:
**Ingestion** (load documents) and **Query** (retrieve + generate).

---

## 1. Ingestion Flow

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
