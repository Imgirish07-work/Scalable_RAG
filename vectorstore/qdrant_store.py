"""Qdrant-backed vector store with dense, sparse, and hybrid (RRF) search."""

import asyncio
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Awaitable, Callable, List, Literal, Optional

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    Prefetch,
    QuantizationSearchParams,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config.settings import settings
from utils.helpers import hash_text
from utils.logger import get_logger
from vectorstore.embeddings import get_embedding_dimension, get_embeddings, _ONNX_PROVIDERS
from vectorstore.base_store import BaseVectorStore

# import lazily-evaluated via try/except so vectorstore stays importable from non-backend contexts
try:
    from backend.metrics import ingest_batch_duration_seconds as _BATCH_DURATION
except ImportError:
    _BATCH_DURATION = None

logger = get_logger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid"]


_SPARSE_MODEL_NAME = "Prithivida/Splade_PP_en_v1"


@lru_cache(maxsize=1)
def _load_sparse_embeddings() -> FastEmbedSparse:
    """Process-wide singleton SPLADE — loaded once, reused across every QdrantStore."""
    kwargs: dict = {}
    local_path = settings.SPLADE_LOCAL_PATH
    if local_path:
        kwargs["specific_model_path"] = local_path
        logger.info("Loading SPLADE from local path: %s", local_path)
    else:
        logger.info("Loading SPLADE from network: %s", _SPARSE_MODEL_NAME)

    sparse = FastEmbedSparse(
        model_name=_SPARSE_MODEL_NAME,
        batch_size=settings.SPLADE_BATCH_SIZE,
        threads=settings.SPLADE_INTRA_OP_THREADS,
        providers=_ONNX_PROVIDERS,
    )
    try:
        active = sparse._model.model.model.get_providers()
        logger.info(
            "SPLADE loaded | requested=%s | active=%s",
            [p if isinstance(p, str) else p[0] for p in _ONNX_PROVIDERS],
            active,
        )
    except Exception:
        logger.info("SPLADE loaded")
    return sparse


class QdrantStore(BaseVectorStore):
    """Qdrant-backed vector store with dense, sparse, and hybrid search."""

    _SPARSE_MODEL = _SPARSE_MODEL_NAME
    _SPARSE_VECTOR_NAME = "sparse"
    _DENSE_VECTOR_NAME = "dense"

    _GRPC_REQUEST_TIMEOUT: int = 10
    _GRPC_KEEPALIVE_TIME_MS: int = 30_000
    _GRPC_KEEPALIVE_TIMEOUT_MS: int = 5_000
    _GRPC_MAX_CONN_AGE_MS: int = 300_000
    _GRPC_MAX_CONN_AGE_GRACE_MS: int = 5_000
    _GRPC_MAX_CONN_IDLE_MS: int = 60_000
    _GRPC_PING_INTERVAL_S: int = 30

    def __init__(
        self,
        collection_name: Optional[str] = None,
        in_memory: bool = True,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        search_mode: SearchMode = "dense",
        client: Optional[QdrantClient] = None,
    ) -> None:
        """Store configuration; no connections are made here."""
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.in_memory = in_memory
        self.search_mode = search_mode
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._injected_client: Optional[QdrantClient] = client

        self._client: Optional[QdrantClient] = None
        self._store: Optional[QdrantVectorStore] = None
        self._sparse_embeddings_instance: Optional[FastEmbedSparse] = None
        self._grpc_active: bool = False
        self._keepalive_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Create the client, ensure the collection exists, and build the store."""
        try:
            self._client = await asyncio.to_thread(self._build_client)

            await asyncio.to_thread(self._create_collection_if_missing)

            self._store = await asyncio.to_thread(self._build_vector_store)

            if self._grpc_active and self._injected_client is None:
                self._keepalive_task = asyncio.create_task(
                    self._keepalive_loop(),
                    name=f"qdrant-keepalive-{self.collection_name}",
                )
                logger.debug(
                    "gRPC keepalive loop started | collection=%s | ping_interval=%ds",
                    self.collection_name, self._GRPC_PING_INTERVAL_S,
                )

            logger.info(
                "QdrantStore ready: collection=%s, mode=%s, search=%s",
                self.collection_name,
                "memory" if self.in_memory else "server",
                self.search_mode,
            )
        except Exception as e:
            logger.exception("Error initializing QdrantStore: %s", e)
            raise

    def _build_client(self) -> QdrantClient:
        """Build and return a QdrantClient for the configured connection mode."""
        if self._injected_client is not None:
            return self._injected_client

        if self.in_memory:
            logger.info("Initializing in-memory Qdrant client")
            return QdrantClient(":memory:")

        url = self._qdrant_url or settings.qdrant_url
        api_key = self._qdrant_api_key or settings.qdrant_api_key

        kwargs: dict = {"url": url, "timeout": self._GRPC_REQUEST_TIMEOUT}
        if api_key:
            kwargs["api_key"] = api_key

        if settings.QDRANT_PREFER_GRPC:
            grpc_options: dict = {
                "grpc.keepalive_time_ms":              self._GRPC_KEEPALIVE_TIME_MS,
                "grpc.keepalive_timeout_ms":           self._GRPC_KEEPALIVE_TIMEOUT_MS,
                "grpc.keepalive_permit_without_calls": 1,
                "grpc.http2.max_pings_without_data":   0,
                "grpc.max_connection_age_ms":          self._GRPC_MAX_CONN_AGE_MS,
                "grpc.max_connection_age_grace_ms":    self._GRPC_MAX_CONN_AGE_GRACE_MS,
                "grpc.max_connection_idle_ms":         self._GRPC_MAX_CONN_IDLE_MS,
            }
            try:
                client = QdrantClient(**kwargs, prefer_grpc=True, grpc_options=grpc_options)
                client.get_collections()
                self._grpc_active = True
                logger.info(
                    "QdrantStore: mode=server, transport=gRPC | url=%s | "
                    "timeout=%ds | keepalive=%dms | max_conn_age=%dms",
                    url,
                    self._GRPC_REQUEST_TIMEOUT,
                    self._GRPC_KEEPALIVE_TIME_MS,
                    self._GRPC_MAX_CONN_AGE_MS,
                )
                return client
            except Exception as exc:
                self._grpc_active = False
                logger.warning(
                    "gRPC connection failed (%s) — falling back to HTTP. "
                    "Set QDRANT_PREFER_GRPC=false to suppress this warning.",
                    exc,
                )

        self._grpc_active = False
        logger.info(
            "QdrantStore: mode=server, transport=HTTP | url=%s | timeout=%ds",
            url, self._GRPC_REQUEST_TIMEOUT,
        )
        return QdrantClient(**kwargs)

    async def _keepalive_loop(self) -> None:
        """Ping Qdrant periodically and reconnect on failure."""
        while True:
            await asyncio.sleep(self._GRPC_PING_INTERVAL_S)
            if self._client is None:
                return
            try:
                await asyncio.to_thread(self._client.get_collections)
                logger.debug("Qdrant gRPC ping OK | collection=%s", self.collection_name)
            except Exception as exc:
                logger.warning(
                    "Qdrant gRPC ping failed | collection=%s | error=%s: %s — reconnecting",
                    self.collection_name, type(exc).__name__, str(exc)[:120],
                )
                await self._reconnect()

    async def _reconnect(self) -> None:
        """Rebuild the gRPC channel and LangChain store wrapper after a drop."""
        try:
            new_client = await asyncio.to_thread(self._build_client)
            old_client = self._client

            self._client = new_client
            self._store = await asyncio.to_thread(self._build_vector_store)

            logger.info(
                "Qdrant gRPC reconnect successful | collection=%s",
                self.collection_name,
            )

            if old_client is not None:
                try:
                    await asyncio.to_thread(old_client.close)
                except Exception:
                    pass

        except Exception as exc:
            logger.error(
                "Qdrant gRPC reconnect failed | collection=%s | error=%s: %s "
                "— will retry in %ds",
                self.collection_name, type(exc).__name__, str(exc)[:120],
                self._GRPC_PING_INTERVAL_S,
            )

    def _create_collection_if_missing(self) -> None:
        """Create the Qdrant collection with the correct vector config for the search mode."""
        try:
            existing = [c.name for c in self._client.get_collections().collections]

            if self.collection_name in existing:
                logger.info(
                    "Collection '%s' already exists — skipping creation",
                    self.collection_name,
                )
                self._validate_collection_config()
                self._ensure_quantization()
                self._ensure_payload_indexes()
                return

            vectors_config = {}
            sparse_vectors_config = {}

            if self.search_mode in ("dense", "hybrid"):
                vectors_config[self._DENSE_VECTOR_NAME] = VectorParams(
                    size=get_embedding_dimension(),
                    distance=Distance.COSINE,
                )

            if self.search_mode in ("sparse", "hybrid"):
                sparse_vectors_config[self._SPARSE_VECTOR_NAME] = SparseVectorParams()

            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config or None,
                sparse_vectors_config=sparse_vectors_config or None,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )

            logger.info(
                "Created collection '%s': search_mode=%s",
                self.collection_name,
                self.search_mode,
            )

            self._ensure_payload_indexes()

        except Exception as e:
            logger.exception("Error creating collection: %s", e)
            raise

    _REQUIRED_PAYLOAD_INDEXES: tuple[str, ...] = (
        "metadata.chunk_id",
        "metadata.user_id",
        "metadata.collection",
        "metadata.doc_id",
    )

    def _ensure_payload_indexes(self) -> None:
        """Create required keyword payload indexes; idempotent and self-healing."""
        for field in self._REQUIRED_PAYLOAD_INDEXES:
            try:
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
                # qdrant returns success on existing indexes, so we can't tell creates
                # from no-ops — log at DEBUG to avoid noise on every QdrantStore init.
                logger.debug(
                    "Payload index ensured | collection=%s | field=%s",
                    self.collection_name, field,
                )
            except Exception as exc:
                logger.debug(
                    "Payload index skipped | collection=%s | field=%s | reason=%s",
                    self.collection_name, field, type(exc).__name__,
                )

    def _validate_collection_config(self) -> None:
        """Warn if the existing collection config does not match the current search mode."""
        try:
            info = self._client.get_collection(self.collection_name)

            if self.search_mode in ("dense", "hybrid"):
                has_dense = (
                    isinstance(info.config.params.vectors, dict)
                    and self._DENSE_VECTOR_NAME in info.config.params.vectors
                )
                if not has_dense:
                    logger.warning(
                        "Collection '%s' may lack dense vectors for search_mode='%s'. "
                        "Consider recreating the collection.",
                        self.collection_name,
                        self.search_mode,
                    )

        except Exception:
            logger.debug(
                "Could not validate collection config for '%s'",
                self.collection_name,
            )

    def _ensure_quantization(self) -> None:
        """Apply Scalar Quantization (int8) to an existing collection if not already set."""
        try:
            info = self._client.get_collection(self.collection_name)
            if info.config.quantization_config is not None:
                logger.debug(
                    "Quantization already configured on '%s' — skipping update",
                    self.collection_name,
                )
                return

            self._client.update_collection(
                collection_name=self.collection_name,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                ),
            )
            logger.info(
                "Scalar Quantization (int8, quantile=0.99, always_ram=True) "
                "applied to existing collection '%s' — re-quantization runs in background",
                self.collection_name,
            )
        except Exception:
            logger.warning(
                "Could not apply quantization to '%s' — search continues unquantized",
                self.collection_name,
                exc_info=True,
            )

    def _build_vector_store(self) -> QdrantVectorStore:
        """Build the LangChain QdrantVectorStore wrapper for the selected search mode."""
        valid_modes = ("dense", "sparse", "hybrid")
        if self.search_mode not in valid_modes:
            raise ValueError(
                f"Invalid search_mode '{self.search_mode}'. Must be one of {valid_modes}."
            )

        try:
            mode_map = {
                "dense": RetrievalMode.DENSE,
                "sparse": RetrievalMode.SPARSE,
                "hybrid": RetrievalMode.HYBRID,
            }

            store_kwargs = {
                "client": self._client,
                "collection_name": self.collection_name,
                "retrieval_mode": mode_map[self.search_mode],
            }

            if self.search_mode in ("dense", "hybrid"):
                store_kwargs["embedding"] = get_embeddings()
                store_kwargs["vector_name"] = self._DENSE_VECTOR_NAME

            if self.search_mode in ("sparse", "hybrid"):
                store_kwargs["sparse_embedding"] = self._get_sparse_embeddings()
                store_kwargs["sparse_vector_name"] = self._SPARSE_VECTOR_NAME

            store = QdrantVectorStore(**store_kwargs)

            logger.debug(
                "QdrantVectorStore built: mode=%s, collection=%s",
                self.search_mode,
                self.collection_name,
            )
            return store

        except Exception as e:
            logger.exception("Error building QdrantVectorStore: %s", e)
            raise

    def _get_sparse_embeddings(self) -> FastEmbedSparse:
        """Return the process-wide SPLADE singleton."""
        if self._sparse_embeddings_instance is None:
            self._sparse_embeddings_instance = _load_sparse_embeddings()
        return self._sparse_embeddings_instance

    async def add_documents(
        self,
        documents: List[Document],
        on_batch_progress: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> List[str]:
        """Embed and store documents with deduplication and outer batching."""
        if not documents:
            logger.warning("add_documents received empty list")
            return []

        try:
            # enrich first (char_count uses original page_content), then dedup, then swap to embed_content
            enriched_docs = self._enrich_metadata(documents)

            new_docs, skipped = await self._filter_existing_documents(enriched_docs)

            if not new_docs:
                logger.info(
                    "All %d chunks already stored — skipping embedding entirely",
                    len(documents),
                )
                return []

            embed_docs = self._prepare_for_embedding(new_docs)

            batch_size  = settings.INGESTION_BATCH_SIZE
            total       = len(embed_docs)
            all_ids: List[str] = []
            failed_batches = 0

            logger.info(
                "Ingestion started: total=%d new chunks, batch_size=%d, "
                "batches=%d, dedup_skipped=%d",
                total,
                batch_size,
                -(-total // batch_size),
                skipped,
            )

            if on_batch_progress is not None:
                await on_batch_progress(0, total)

            for batch_start in range(0, total, batch_size):
                batch      = embed_docs[batch_start : batch_start + batch_size]
                batch_num  = batch_start // batch_size + 1
                batch_end  = min(batch_start + batch_size, total)

                try:
                    _t0 = time.perf_counter()
                    batch_ids = await asyncio.to_thread(
                        self._store.add_documents,
                        batch,
                        batch_size=settings.SPLADE_BATCH_SIZE,
                    )
                    _t1 = time.perf_counter()
                    if _BATCH_DURATION is not None:
                        _BATCH_DURATION.observe(_t1 - _t0)
                    all_ids.extend(batch_ids)
                    logger.info(
                        "Ingestion batch %d/%d complete: chunks=%d/%d "
                        "| committed=%d | remaining=%d | elapsed=%.1fs",
                        batch_num,
                        -(-total // batch_size),
                        batch_end,
                        total,
                        len(all_ids),
                        total - batch_end,
                        _t1 - _t0,
                    )
                    if on_batch_progress is not None:
                        await on_batch_progress(batch_end, total)
                except Exception as batch_err:
                    failed_batches += 1
                    logger.error(
                        "Ingestion batch %d/%d FAILED (chunks %d-%d): %s "
                        "— %d chunks already committed are preserved.",
                        batch_num,
                        -(-total // batch_size),
                        batch_start + 1,
                        batch_end,
                        batch_err,
                        len(all_ids),
                    )
                    raise

            logger.info(
                "QdrantStore ingestion complete: stored=%d, dedup_skipped=%d",
                len(all_ids),
                skipped,
            )
            return all_ids

        except Exception as e:
            logger.exception("Error in add_documents: %s", e)
            raise

    async def _filter_existing_documents(
        self,
        documents: List[Document],
    ) -> tuple[List[Document], int]:
        """Filter out chunks already stored using a single batch scroll query."""
        if not documents:
            return documents, 0

        chunk_ids = [
            doc.metadata.get("chunk_id") or hash_text(doc.page_content)
            for doc in documents
        ]

        try:
            existing_points, _ = await asyncio.to_thread(
                self._client.scroll,
                collection_name=self.collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.chunk_id",
                            match=MatchAny(any=chunk_ids),
                        )
                    ]
                ),
                with_payload=["metadata.chunk_id"],
                limit=len(chunk_ids),
            )

            existing_ids = {
                point.payload.get("metadata", {}).get("chunk_id")
                for point in existing_points
            }

            new_docs = [
                doc for doc, cid in zip(documents, chunk_ids)
                if cid not in existing_ids
            ]

            skipped = len(documents) - len(new_docs)
            if skipped > 0:
                logger.info(
                    "Deduplication: skipping %d/%d chunks already in Qdrant",
                    skipped, len(documents),
                )

            return new_docs, skipped

        except Exception as exc:
            logger.warning(
                "Dedup check failed, ingesting all %d chunks: %s",
                len(documents), exc,
            )
            return documents, 0

    def _prepare_for_embedding(self, documents: List[Document]) -> List[Document]:
        """Swap page_content with embed_content so LangChain embeds the richer text."""
        embed_docs = []

        for doc in documents:
            embed_content = doc.metadata.get("embed_content", doc.page_content)

            metadata = doc.metadata.copy()
            metadata["original_content"] = doc.page_content

            embed_docs.append(
                Document(page_content=embed_content, metadata=metadata)
            )

        return embed_docs

    def _enrich_metadata(self, documents: List[Document]) -> List[Document]:
        """Attach required payload fields to every document before storage."""
        total_chunks = len(documents)
        enriched_docs = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata.copy()

            metadata.setdefault("doc_id", "")
            metadata.setdefault("user_id", "")
            metadata.setdefault("collection", "default")

            metadata.setdefault("source", "unknown")
            metadata.setdefault("page", 0)

            metadata.setdefault("chunk_index", i)
            metadata.setdefault("total_chunks", total_chunks)

            metadata["char_count"] = len(doc.page_content)
            metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()

            enriched_docs.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )

        return enriched_docs

    async def similarity_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
        filter_user_id: Optional[str] = None,
        filter_collection: Optional[str] = None,
    ) -> List[Document]:
        """Search for semantically similar documents."""
        if not query or not query.strip():
            logger.warning("similarity_search received empty query")
            return []

        if k <= 0:
            return []

        try:
            qdrant_filter = self._build_filter(filter_user_id, filter_collection)

            if score_threshold is not None:
                results = await self._search_with_scores(
                    query, k, score_threshold, qdrant_filter
                )
            else:
                results = await self._search_without_scores(
                    query, k, qdrant_filter
                )

            results = self._restore_original_content(results)

            return results

        except Exception as e:
            logger.exception("Error in similarity_search: %s", e)
            raise

    async def _search_with_scores(
        self,
        query: str,
        k: int,
        score_threshold: float,
        qdrant_filter: Optional[Filter],
    ) -> List[Document]:
        """Search with relevance score filtering and score attachment."""
        results = await asyncio.to_thread(
            self._store.similarity_search_with_relevance_scores,
            query=query,
            k=k,
            filter=qdrant_filter,
        )

        relevant_results = []
        for doc, score in results:
            if score >= score_threshold:
                doc.metadata["relevance_score"] = round(score, 4)
                relevant_results.append(doc)

        logger.debug(
            "similarity_search: query='%s', results=%d, threshold=%.2f, filtered_out=%d",
            query[:50],
            len(relevant_results),
            score_threshold,
            len(results) - len(relevant_results),
        )

        return relevant_results

    async def _search_without_scores(
        self,
        query: str,
        k: int,
        qdrant_filter: Optional[Filter],
    ) -> List[Document]:
        """Search without score filtering — returns top-k directly."""
        results = await asyncio.to_thread(
            self._store.similarity_search,
            query=query,
            k=k,
            filter=qdrant_filter,
        )

        logger.debug(
            "similarity_search: query='%s', results=%d",
            query[:50],
            len(results),
        )

        return results

    def _restore_original_content(self, documents: List[Document]) -> List[Document]:
        """Restore clean page_content after retrieval, reversing the embed_content swap."""
        restored = []
        for doc in documents:
            original = doc.metadata.get("original_content")
            if original is not None:
                doc = Document(page_content=original, metadata=doc.metadata)
            restored.append(doc)

        return restored

    def _build_filter(
        self,
        user_id: Optional[str],
        collection: Optional[str] = None,
    ) -> Optional[Filter]:
        """Build a Qdrant payload filter for tenant + logical-collection scoping."""
        if user_id is not None and not user_id:
            raise ValueError(
                "user_id was passed as empty string; pass None for explicit admin "
                "queries or a real user_id for tenant-scoped queries"
            )

        conditions: list[FieldCondition] = []

        if user_id:
            conditions.append(
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id),
                )
            )

        if collection:
            conditions.append(
                FieldCondition(
                    key="metadata.collection",
                    match=MatchValue(value=collection),
                )
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    async def similarity_search_with_vectors(
        self,
        query: str,
        k: int,
        filter_user_id: Optional[str] = None,
        filter_collection: Optional[str] = None,
    ) -> List[Document]:
        """Search top-k and return stored dense embedding vectors in metadata['vector']."""
        if not query or not query.strip():
            logger.warning("similarity_search_with_vectors received empty query")
            return []
        if k <= 0:
            return []

        try:
            
            embeddings_model = await asyncio.to_thread(get_embeddings)
            query_vector = await asyncio.to_thread(
                embeddings_model.embed_query, query
            )

            qdrant_filter = self._build_filter(filter_user_id, filter_collection)

            # rescore=True re-ranks int8 ANN candidates with float32 vectors to recover SQ recall loss
            response = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                using=self._DENSE_VECTOR_NAME,
                limit=k,
                with_vectors=True,
                with_payload=True,
                query_filter=qdrant_filter,
                search_params=SearchParams(
                    quantization=QuantizationSearchParams(rescore=True)
                ),
            )

            docs = []
            for point in response.points:
                payload = point.payload or {}
                page_content = payload.get("page_content", "")
                metadata = dict(payload.get("metadata", {}))

                metadata["relevance_score"] = round(float(point.score), 4)

                raw_vec = point.vector
                if isinstance(raw_vec, dict):
                    metadata["vector"] = raw_vec.get(self._DENSE_VECTOR_NAME)
                else:
                    metadata["vector"] = raw_vec

                docs.append(Document(page_content=page_content, metadata=metadata))

            docs = self._restore_original_content(docs)

            logger.debug(
                "similarity_search_with_vectors: query='%s', results=%d",
                query[:50],
                len(docs),
            )

            return docs

        except Exception as e:
            logger.exception("Error in similarity_search_with_vectors: %s", e)
            raise

    async def hybrid_search_with_vectors(
        self,
        query: str,
        k: int,
        filter_user_id: Optional[str] = None,
        filter_collection: Optional[str] = None,
    ) -> List[Document]:
        """Hybrid RRF search returning top-k results with dense embedding vectors."""
        if not query or not query.strip():
            logger.warning("hybrid_search_with_vectors received empty query")
            return []
        if k <= 0:
            return []

        try:
            embeddings_model, sparse_model = await asyncio.gather(
                asyncio.to_thread(get_embeddings),
                asyncio.to_thread(self._get_sparse_embeddings),
            )

            query_vector, sparse_vector = await asyncio.gather(
                asyncio.to_thread(embeddings_model.embed_query, query),
                asyncio.to_thread(sparse_model.embed_query, query),
            )

            qdrant_filter = self._build_filter(filter_user_id, filter_collection)

            coarse_k = max(k * 3, 20)

            response = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using=self._DENSE_VECTOR_NAME,
                        limit=coarse_k,
                        params=SearchParams(
                            quantization=QuantizationSearchParams(rescore=True)
                        ),
                    ),
                    Prefetch(
                        query=SparseVector(
                            indices=sparse_vector.indices,
                            values=sparse_vector.values,
                        ),
                        using=self._SPARSE_VECTOR_NAME,
                        limit=coarse_k,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=k,
                with_vectors=True,
                with_payload=True,
                query_filter=qdrant_filter,
            )

            docs = []
            for point in response.points:
                payload = point.payload or {}
                page_content = payload.get("page_content", "")
                metadata = dict(payload.get("metadata", {}))

                metadata["relevance_score"] = round(float(point.score), 4)

                raw_vec = point.vector
                if isinstance(raw_vec, dict):
                    metadata["vector"] = raw_vec.get(self._DENSE_VECTOR_NAME)
                else:
                    metadata["vector"] = raw_vec

                docs.append(Document(page_content=page_content, metadata=metadata))

            docs = self._restore_original_content(docs)

            logger.debug(
                "hybrid_search_with_vectors: query='%s', results=%d",
                query[:50],
                len(docs),
            )

            return docs

        except Exception as e:
            logger.exception("Error in hybrid_search_with_vectors: %s", e)
            raise

    async def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete every point whose payload.metadata.doc_id matches; idempotent."""
        match_filter = Filter(
            must=[FieldCondition(
                key="metadata.doc_id",
                match=MatchValue(value=doc_id),
            )]
        )
        try:
            await asyncio.to_thread(
                self._client.delete,
                collection_name=self.collection_name,
                points_selector=match_filter,
                wait=True,
            )
            logger.info(
                "Qdrant points deleted | collection=%s | doc_id=%s",
                self.collection_name, doc_id,
            )
        except Exception as exc:
            if "doesn't exist" in str(exc).lower() or "not found" in str(exc).lower():
                logger.info(
                    "Qdrant delete skipped — collection does not exist yet "
                    "| collection=%s | doc_id=%s",
                    self.collection_name, doc_id,
                )
                return
            logger.exception(
                "delete_by_doc_id failed | collection=%s | doc_id=%s",
                self.collection_name, doc_id,
            )
            raise

    async def delete_collection(self) -> None:
        """Permanently delete the entire Qdrant collection."""
        try:
            await asyncio.to_thread(
                self._client.delete_collection,
                collection_name=self.collection_name,
            )
            logger.warning("Collection '%s' deleted", self.collection_name)
        except Exception as e:
            logger.exception("Error deleting collection: %s", e)
            raise

    async def list_all_collections(self) -> List[str]:
        """Return every collection name visible in the connected Qdrant cluster."""
        raw = await asyncio.to_thread(self._client.get_collections)
        return [c.name for c in raw.collections]

    async def get_collection_stats(self) -> dict:
        """Return collection statistics for observability dashboards."""
        try:
            info = await asyncio.to_thread(
                self._client.get_collection,
                collection_name=self.collection_name,
            )
            return {
                "backend": "qdrant",
                "collection_name": self.collection_name,
                "document_count": info.points_count,
                "embedding_model": settings.embedding_model,
                "search_mode": self.search_mode,
                "mode": "memory" if self.in_memory else "server",
            }
        except Exception as e:
            logger.exception("get_collection_stats failed: %s", e)
            return {}

    async def close(self) -> None:
        """Close the Qdrant client connection and release all held references."""
        if self._keepalive_task is not None and not self._keepalive_task.done():
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None

        if self._client:
            try:
                await asyncio.to_thread(self._client.close)
                logger.info("QdrantStore connection closed")
            except Exception as e:
                logger.exception("Error closing QdrantStore: %s", e)

        self._client = None
        self._store = None
