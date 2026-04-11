"""
Qdrant vector store — concrete implementation of BaseVectorStore.

Design:
    Subclasses BaseVectorStore and provides dense, sparse, and hybrid search
    backed by Qdrant. A single QdrantClient is shared across all operations.
    All sync Qdrant/LangChain calls are wrapped in asyncio.to_thread() to keep
    the FastAPI event loop non-blocking.

    Three connection modes (switched via env vars, no code changes required):
        Mode 1 → in_memory=True                   — dev / unit tests
        Mode 2 → in_memory=False, local Docker     — staging
        Mode 3 → in_memory=False, Qdrant Cloud URL — production

    Three search modes:
        dense  — semantic similarity via BGE embeddings
        sparse — keyword matching via SPLADE
        hybrid — dense + sparse RRF fusion (recommended; best recall)

    Embedding strategy:
        Uses metadata['embed_content'] for embedding when available.
        embed_content = "Title: {filename} | Section: {section}\\n{content}"
        Produces richer vectors with document context. Original page_content
        is preserved untouched for LLM consumption.

    Payload schema per stored chunk:
        doc_id, user_id, source, page, chunk_index,
        total_chunks, char_count, ingested_at

Chain of Responsibility:
    Called by the RAG pipeline (retrievers, ingestion service).
    QdrantStore.initialize() → builds client → creates collection → builds
    LangChain store wrapper. add_documents() is called by the ingestion
    pipeline; similarity_search() and its variants are called by retrievers.

Dependencies:
    qdrant_client, langchain_qdrant, langchain_core, config.settings,
    vectorstore.embeddings, vectorstore.base_store
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import List, Literal, Optional

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

logger = get_logger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid"]


class QdrantStore(BaseVectorStore):
    """Qdrant-backed vector store with dense, sparse, and hybrid search.

    Uses a two-phase initialisation pattern: the constructor stores config
    only, and initialize() establishes the actual connection and collection.
    This allows the object to be constructed synchronously and then initialised
    inside an async context.

    Attributes:
        collection_name: Qdrant collection name.
        in_memory: If True, uses an in-memory Qdrant client (dev/testing).
        search_mode: Active search mode ('dense', 'sparse', or 'hybrid').
        _client: Single shared QdrantClient instance.
        _store: LangChain QdrantVectorStore wrapper around _client.
        _sparse_embeddings_instance: Lazy-loaded SPLADE sparse model.
        _injected_client: Optional client provided at construction time.
        _qdrant_url: Qdrant server URL (overrides settings when set).
        _qdrant_api_key: Qdrant API key (overrides settings when set).
    """

    # Vector name constants — must match collection creation and store construction.
    _SPARSE_MODEL = "Prithivida/Splade_PP_en_v1"
    _SPARSE_VECTOR_NAME = "sparse"
    _DENSE_VECTOR_NAME = "dense"

    def __init__(
        self,
        collection_name: Optional[str] = None,
        in_memory: bool = True,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        search_mode: SearchMode = "dense",
        client: Optional[QdrantClient] = None,
    ) -> None:
        """Store configuration; no connections are made here.

        Args:
            collection_name: Qdrant collection name. Defaults to settings value.
            in_memory: If True, use an in-memory Qdrant client (no server needed).
            qdrant_url: Qdrant server URL. Overrides settings when provided.
            qdrant_api_key: Qdrant API key. Overrides settings when provided.
            search_mode: Search strategy — 'dense', 'sparse', or 'hybrid'.
            client: Optional existing QdrantClient to reuse. When provided,
                the caller's client is used as-is, skipping client construction.
                Required when multiple collections must share one in-memory
                database (the same pattern as sharing one server in production).
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.in_memory = in_memory
        self.search_mode = search_mode
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._injected_client: Optional[QdrantClient] = client

        # Populated by initialize() — not here (two-phase pattern).
        self._client: Optional[QdrantClient] = None
        self._store: Optional[QdrantVectorStore] = None
        self._sparse_embeddings_instance: Optional[FastEmbedSparse] = None

    async def initialize(self) -> None:
        """Create the client, ensure the collection exists, and build the store.

        Must be called once before add_documents() or similarity_search().
        Safe to call multiple times (idempotent).

        Raises:
            Exception: If connection or collection creation fails.
        """
        try:
            self._client = self._build_client()

            # Collection creation is sync — offload to thread to avoid blocking.
            await asyncio.to_thread(self._create_collection_if_missing)

            self._store = self._build_vector_store()

            logger.info(
                "QdrantStore ready: collection=%s, mode=%s, search=%s",
                self.collection_name,
                "memory" if self.in_memory else "server",
                self.search_mode,
            )
        except Exception as e:
            logger.exception("Error initializing QdrantStore: %s", e)
            raise

    # Client construction

    def _build_client(self) -> QdrantClient:
        """Build and return a QdrantClient for the configured connection mode.

        If a client was injected at construction time, returns it directly
        (shared client — no new connection is created).

        Transport selection for server mode:
            QDRANT_PREFER_GRPC=True  → gRPC on port 6334 (~24% faster).
                Falls back to HTTP automatically if the gRPC handshake fails
                (e.g., port blocked by corporate firewall).
            QDRANT_PREFER_GRPC=False → HTTP on port 6333 (default).

        Returns:
            Configured QdrantClient instance.
        """
        if self._injected_client is not None:
            return self._injected_client

        if self.in_memory:
            logger.info("Initializing in-memory Qdrant client")
            return QdrantClient(":memory:")

        url = self._qdrant_url or settings.qdrant_url
        api_key = self._qdrant_api_key or settings.qdrant_api_key

        kwargs: dict = {"url": url}
        if api_key:
            kwargs["api_key"] = api_key

        if settings.QDRANT_PREFER_GRPC:
            try:
                client = QdrantClient(**kwargs, prefer_grpc=True)
                # get_collections() forces the gRPC handshake immediately —
                # cheap no-data call that confirms the port is reachable.
                client.get_collections()
                logger.info(
                    "QdrantStore: mode=server, transport=gRPC, url=%s", url
                )
                return client
            except Exception as exc:
                logger.warning(
                    "gRPC connection failed (%s) — falling back to HTTP. "
                    "Set QDRANT_PREFER_GRPC=false to suppress this warning.",
                    exc,
                )

        logger.info("QdrantStore: mode=server, transport=HTTP, url=%s", url)
        return QdrantClient(**kwargs)

    # Collection management

    def _create_collection_if_missing(self) -> None:
        """Create the Qdrant collection with the correct vector config for the search mode.

        Sync — runs inside asyncio.to_thread() from initialize().

        Vector config by search mode:
            dense  → dense vectors only (BGE cosine)
            sparse → sparse vectors only (SPLADE)
            hybrid → both dense + sparse (RRF fusion)

        Collection is created with Scalar Quantization (int8):
            SQ compresses float32 → int8 (4× less RAM, 2-3× faster ANN search).
            quantile=0.99 clips the top 1% of values to reduce outlier impact.
            always_ram=True keeps quantized vectors in RAM for lowest latency.
            rescore=True (at search time) re-ranks the top-k ANN candidates
            with original float32 vectors, recovering the ~1% recall loss.

        Idempotent — skips creation if the collection already exists.
        """
        try:
            existing = [c.name for c in self._client.get_collections().collections]

            if self.collection_name in existing:
                logger.info(
                    "Collection '%s' already exists — skipping creation",
                    self.collection_name,
                )
                # Soft validation — warns on schema mismatch but does not fail.
                self._validate_collection_config()
                # Apply SQ to existing collection if not yet configured.
                self._ensure_quantization()
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

            # Keyword indexes on filtered fields — required on Qdrant Cloud/server.
            # In-memory mode skips enforcement but creating them is harmless.
            for field in ("metadata.chunk_id", "metadata.user_id"):
                self._client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            logger.info(
                "Created payload indexes: collection=%s, fields=[metadata.chunk_id, metadata.user_id]",
                self.collection_name,
            )

        except Exception as e:
            logger.exception("Error creating collection: %s", e)
            raise

    def _validate_collection_config(self) -> None:
        """Warn if the existing collection config does not match the current search mode.

        Soft check — logs a warning but does not raise. Full schema migration
        (e.g., adding sparse vectors to a dense-only collection) requires a
        manual admin step and is outside the scope of this method.

        Sync — called from _create_collection_if_missing().
        """
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
            # Validation is best-effort — do not fail on check errors.
            logger.debug(
                "Could not validate collection config for '%s'",
                self.collection_name,
            )

    def _ensure_quantization(self) -> None:
        """Apply Scalar Quantization (int8) to an existing collection if not already set.

        Called once per startup when the collection already exists.
        Qdrant re-quantizes in the background — no downtime, no data loss.
        Idempotent: skips the update if SQ is already configured.

        Sync — runs inside asyncio.to_thread() via _create_collection_if_missing().
        """
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
            # Quantization is an optimisation — not required for correctness.
            logger.warning(
                "Could not apply quantization to '%s' — search continues unquantized",
                self.collection_name,
                exc_info=True,
            )

    # Vector store construction

    def _build_vector_store(self) -> QdrantVectorStore:
        """Build the LangChain QdrantVectorStore wrapper for the selected search mode.

        Sync — no I/O, pure object construction.

        vector_name and sparse_vector_name must exactly match what was used in
        _create_collection_if_missing(). A mismatch causes silent read/write failures.

        Returns:
            Configured QdrantVectorStore instance.

        Raises:
            ValueError: If search_mode is not one of 'dense', 'sparse', 'hybrid'.
        """
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
        """Return the sparse embedding model, instantiating it on first call.

        Lazy-loaded and cached on self — only created when sparse or hybrid
        mode is actually used. Survives across multiple _build_vector_store calls.

        Returns:
            FastEmbedSparse instance for the configured SPLADE model.
        """
        if self._sparse_embeddings_instance is None:
            try:
                kwargs: dict = {}
                local_path = settings.SPLADE_LOCAL_PATH
                if local_path:
                    kwargs["specific_model_path"] = local_path
                    logger.info(
                        "Loading SPLADE from local path (skipping download): %s",
                        local_path,
                    )
                else:
                    logger.info(
                        "Initializing sparse embedding model (requires network): %s",
                        self._SPARSE_MODEL,
                    )
                self._sparse_embeddings_instance = FastEmbedSparse(
                    model_name=self._SPARSE_MODEL,
                    batch_size=settings.SPLADE_BATCH_SIZE,
                    threads=settings.SPLADE_INTRA_OP_THREADS,
                    providers=_ONNX_PROVIDERS,
                    **kwargs,
                )

                # Log which ONNX execution provider SPLADE is actually using.
                # FastEmbedSparse._model → SparseTextEmbedding.model → SpladePP.model → ort.InferenceSession
                try:
                    splade_session = self._sparse_embeddings_instance._model.model.model
                    active_providers = splade_session.get_providers()
                    logger.info(
                        "SPLADE sparse model loaded | requested=%s | active=%s",
                        [p if isinstance(p, str) else p[0] for p in _ONNX_PROVIDERS],
                        active_providers,
                    )
                except Exception:
                    logger.info("Sparse embedding model initialized successfully")
            except Exception as e:
                logger.exception("Error initializing sparse embeddings: %s", e)
                raise

        return self._sparse_embeddings_instance

    # Write — add documents

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents in Qdrant with deduplication and outer batch processing.

        Embedding strategy:
            Uses metadata['embed_content'] if available — this contains
            "Title: {file} | Section: {heading}\\n{content}" for richer
            semantic vectors. Falls back to page_content when not set.
            Original page_content is saved in metadata['original_content']
            so similarity_search() can restore it for LLM consumption.

        Batching strategy:
            Dedup runs once upfront via a single Qdrant scroll (efficient).
            New docs are split into INGESTION_BATCH_SIZE outer batches.
            Each batch: LangChain sub-batches into SPLADE_BATCH_SIZE chunks
            so the SPLADE MLM tensor fits within VRAM budget → upsert → log.
            Batches already committed survive a failure in a later batch.

            SPLADE VRAM budget:
                SPLADE_BATCH_SIZE=16 → (16, 512, 30522)×4B = 1.0 GB < 4 GB VRAM
                default 64           → (64, 512, 30522)×4B = 4.0 GB — borderline
                100                  → 6.27 GB — exceeds VRAM, triggers silent CPU fallback

        Args:
            documents: List of Documents from the Chunker.

        Returns:
            List of assigned Qdrant point IDs from all committed batches.

        Raises:
            Exception: If embedding or storage fails for any batch.
        """
        if not documents:
            logger.warning("add_documents received empty list")
            return []

        try:
            # Order matters: enrich first (char_count uses original page_content),
            # then dedup, then swap to embed_content. Do NOT reorder these calls.
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
                -(-total // batch_size),   # ceiling division
                skipped,
            )

            for batch_start in range(0, total, batch_size):
                batch      = embed_docs[batch_start : batch_start + batch_size]
                batch_num  = batch_start // batch_size + 1
                batch_end  = min(batch_start + batch_size, total)

                try:
                    _t0 = time.perf_counter()
                    # batch_size=SPLADE_BATCH_SIZE controls LangChain's internal
                    # _generate_rest_batches, keeping the SPLADE MLM output tensor
                    # within VRAM limits (see SPLADE VRAM budget in docstring above).
                    batch_ids = await asyncio.to_thread(
                        self._store.add_documents,
                        batch,
                        batch_size=settings.SPLADE_BATCH_SIZE,
                    )
                    _t1 = time.perf_counter()
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
        """Filter out chunks already stored in Qdrant using a single batch scroll query.

        The Chunker stores chunk_id = hash_text(page_content) in every chunk's
        metadata. We use a single MatchAny scroll query — not N individual queries —
        to check all chunk_ids at once, avoiding wasted embedding work on duplicates.

        Falls back to ingesting all documents if the dedup check fails, so
        a transient Qdrant error never blocks ingestion.

        Args:
            documents: Enriched documents from _enrich_metadata().

        Returns:
            Tuple of (new_documents_only, skipped_count).
        """
        if not documents:
            return documents, 0

        # Fall back to hashing page_content when chunk_id is missing
        # (e.g., documents ingested outside the standard Chunker pipeline).
        chunk_ids = [
            doc.metadata.get("chunk_id") or hash_text(doc.page_content)
            for doc in documents
        ]

        try:
            # Single MatchAny scroll — OR across all chunk_ids in one request.
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
            # Never block ingestion on a dedup failure.
            logger.warning(
                "Dedup check failed, ingesting all %d chunks: %s",
                len(documents), exc,
            )
            return documents, 0

    def _prepare_for_embedding(self, documents: List[Document]) -> List[Document]:
        """Swap page_content with embed_content so LangChain embeds the richer text.

        LangChain QdrantVectorStore embeds whatever is in page_content. The
        Chunker places richer text in metadata['embed_content'] that includes
        title and section context. We swap it in here so the vector captures
        that context.

        Original page_content is saved in metadata['original_content'] so
        similarity_search() can restore clean text for LLM consumption.

        Sync — pure data transformation, no I/O.

        Args:
            documents: Enriched documents ready for storage.

        Returns:
            New Document list with embed_content as page_content.
        """
        embed_docs = []

        for doc in documents:
            embed_content = doc.metadata.get("embed_content", doc.page_content)

            # Preserve original clean text for LLM retrieval.
            metadata = doc.metadata.copy()
            metadata["original_content"] = doc.page_content

            embed_docs.append(
                Document(page_content=embed_content, metadata=metadata)
            )

        return embed_docs

    def _enrich_metadata(self, documents: List[Document]) -> List[Document]:
        """Attach required payload fields to every document before storage.

        Caller-supplied doc_id and user_id are preserved via setdefault.
        char_count and ingested_at are always overwritten for accuracy.

        Sync — pure data transformation, no I/O.

        Args:
            documents: Raw documents from Chunker.

        Returns:
            Documents with enriched metadata ready for Qdrant storage.
        """
        total_chunks = len(documents)
        enriched_docs = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata.copy()

            # Placeholders — replaced by real UUIDs after the auth layer is wired up.
            metadata.setdefault("doc_id", "")
            metadata.setdefault("user_id", "")

            metadata.setdefault("source", "unknown")
            metadata.setdefault("page", 0)

            metadata.setdefault("chunk_index", i)
            metadata.setdefault("total_chunks", total_chunks)

            # Always overwritten to reflect the current ingest run.
            metadata["char_count"] = len(doc.page_content)
            metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()

            enriched_docs.append(
                Document(page_content=doc.page_content, metadata=metadata)
            )

        return enriched_docs

    # Read — similarity search

    async def similarity_search(
        self,
        query: str,
        k: int = 3,
        score_threshold: Optional[float] = None,
        filter_user_id: Optional[str] = None,
    ) -> List[Document]:
        """Search for semantically similar documents.

        Returns Documents with clean page_content (original text, not the
        embed_content prefix). When score_threshold is provided, relevance
        scores are attached to metadata['relevance_score'].

        Score threshold guide (BGE + cosine, dense mode):
            >= 0.7 — very confident match
            >= 0.5 — good match (recommended default)
            >= 0.3 — loose match
             < 0.3 — likely irrelevant

        Args:
            query: Search text to embed and compare.
            k: Number of results to return.
            score_threshold: Minimum similarity score (0.0-1.0).
                             None = return all top-k unfiltered.
            filter_user_id: Filter to a specific user's documents.
                            None = search entire collection.

        Returns:
            List of matching Documents with clean page_content.

        Raises:
            Exception: If search fails.
        """
        if not query or not query.strip():
            logger.warning("similarity_search received empty query")
            return []

        if k <= 0:
            return []

        try:
            qdrant_filter = self._build_filter(filter_user_id)

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
        """Search with relevance score filtering and score attachment.

        Scores are preserved in metadata['relevance_score'] so the RAG layer
        can use them for reranking or confidence decisions.

        Args:
            query: Search query text.
            k: Number of results.
            score_threshold: Minimum score to include in results.
            qdrant_filter: Optional Qdrant payload filter.

        Returns:
            Filtered list of Documents with scores in metadata.
        """
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
        """Search without score filtering — returns top-k directly.

        Args:
            query: Search query text.
            k: Number of results.
            qdrant_filter: Optional Qdrant payload filter.

        Returns:
            List of top-k Documents.
        """
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
        """Restore clean page_content after retrieval, reversing the embed_content swap.

        During add_documents(), page_content was replaced with embed_content for
        richer embeddings. This method swaps back the original text so the LLM
        receives clean content without the "Title: ... | Section: ..." prefix.

        Documents that lack original_content (old data or external ingestion)
        are returned unchanged.

        Sync — pure data transformation, no I/O.

        Args:
            documents: Documents returned from a Qdrant search.

        Returns:
            Documents with clean page_content for LLM consumption.
        """
        restored = []
        for doc in documents:
            original = doc.metadata.get("original_content")
            if original is not None:
                doc = Document(page_content=original, metadata=doc.metadata)
            restored.append(doc)

        return restored

    def _build_filter(self, user_id: Optional[str]) -> Optional[Filter]:
        """Build a Qdrant payload filter for per-user document isolation.

        Args:
            user_id: User ID to filter by. None returns no filter.

        Returns:
            Qdrant Filter object, or None for unfiltered search.
        """
        if user_id is None:
            return None

        return Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=user_id),
                )
            ]
        )

    async def similarity_search_with_vectors(
        self,
        query: str,
        k: int,
        filter_user_id: Optional[str] = None,
    ) -> List[Document]:
        """Search for top-k results and return stored dense embedding vectors alongside them.

        Uses QdrantClient.query_points(with_vectors=True) directly instead of the
        LangChain wrapper, so each result carries its stored dense vector in
        metadata['vector']. This vector is used by ContextRanker._rank_mmr() for
        inter-chunk cosine similarity — eliminating the need to re-embed chunks
        on every query (~2-4s saved per search).

        Content and score behaviour is identical to similarity_search():
            - page_content is restored to the original clean text.
            - relevance_score is attached to metadata['relevance_score'].
            - rescore=True re-ranks int8 ANN candidates with float32 vectors.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter_user_id: Filter to a specific user's documents.
                            None = search entire collection.

        Returns:
            List of Documents with clean page_content, relevance_score in metadata,
            and the dense embedding vector in metadata['vector'].

        Raises:
            Exception: If search fails.
        """
        if not query or not query.strip():
            logger.warning("similarity_search_with_vectors received empty query")
            return []
        if k <= 0:
            return []

        try:
            embeddings_model = get_embeddings()
            query_vector = await asyncio.to_thread(
                embeddings_model.embed_query, query
            )

            qdrant_filter = self._build_filter(filter_user_id)

            # query_points replaces the deprecated client.search() (Qdrant client v1.7+).
            # rescore=True: re-ranks int8 ANN results with float32 vectors — recovers SQ recall loss.
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

                # Extract the dense vector from the named-vector dict.
                raw_vec = point.vector
                if isinstance(raw_vec, dict):
                    metadata["vector"] = raw_vec.get(self._DENSE_VECTOR_NAME)
                else:
                    metadata["vector"] = raw_vec  # fallback for unnamed vector format

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
    ) -> List[Document]:
        """Hybrid RRF search returning top-k results with dense embedding vectors.

        Uses Qdrant's native prefetch + RRF fusion in a single query_points call.
        Dense and sparse searches run in parallel inside Qdrant, results are fused
        via Reciprocal Rank Fusion, and dense vectors are returned alongside payloads
        for downstream MMR diversity scoring.

        Dense leg: rescore=True re-ranks int8 ANN candidates with float32 vectors
        before RRF fusion — recovers recall loss from scalar quantization.
        Sparse leg: SPLADE vectors are not quantized — no rescore params needed.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter_user_id: Filter to a specific user's documents.
                            None = search entire collection.

        Returns:
            List of Documents with clean page_content, relevance_score in metadata,
            and the dense embedding vector in metadata['vector'].

        Raises:
            Exception: If hybrid search fails.
        """
        if not query or not query.strip():
            logger.warning("hybrid_search_with_vectors received empty query")
            return []
        if k <= 0:
            return []

        try:
            # Compute dense and sparse embeddings in parallel — independent operations.
            embeddings_model = get_embeddings()
            sparse_model = self._get_sparse_embeddings()

            query_vector, sparse_vector = await asyncio.gather(
                asyncio.to_thread(embeddings_model.embed_query, query),
                asyncio.to_thread(sparse_model.embed_query, query),
            )

            qdrant_filter = self._build_filter(filter_user_id)

            # Fetch more candidates per leg so RRF has a large enough pool to fuse.
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

                # Extract dense vector for MMR inter-chunk diversity scoring.
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

    # Admin operations

    async def delete_collection(self) -> None:
        """Permanently delete the entire Qdrant collection.

        Irreversible — all vectors and metadata are lost. Use with caution.

        Raises:
            Exception: If deletion fails.
        """
        try:
            await asyncio.to_thread(
                self._client.delete_collection,
                collection_name=self.collection_name,
            )
            logger.warning("Collection '%s' deleted", self.collection_name)
        except Exception as e:
            logger.exception("Error deleting collection: %s", e)
            raise

    async def get_collection_stats(self) -> dict:
        """Return collection statistics for observability dashboards.

        Returns:
            Dict containing backend, collection_name, document_count,
            embedding_model, search_mode, and memory/server mode.
            Returns an empty dict on failure (never raises).
        """
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
        """Close the Qdrant client connection and release all held references.

        Prevents connection leaks on application shutdown. Safe to call
        multiple times.

        Usage in FastAPI:
            @app.on_event("shutdown")
            async def shutdown():
                await store.close()
        """
        if self._client:
            try:
                await asyncio.to_thread(self._client.close)
                logger.info("QdrantStore connection closed")
            except Exception as e:
                logger.exception("Error closing QdrantStore: %s", e)

        # Release references to allow garbage collection.
        self._client = None
        self._store = None
