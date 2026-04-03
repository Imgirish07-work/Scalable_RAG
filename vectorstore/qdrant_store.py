"""
Qdrant vector store — supports dense, sparse, and hybrid search.

Single QdrantClient shared across all operations. All sync Qdrant/LangChain
calls are wrapped in asyncio.to_thread() to keep the event loop non-blocking.

Three connection modes (switch via env vars, zero code change):
    Mode 1 → in_memory=True                   (dev, testing)
    Mode 2 → in_memory=False, local Docker     (staging)
    Mode 3 → in_memory=False, Qdrant Cloud URL (production)

Three search modes:
    dense  → semantic similarity using BGE embeddings
    sparse → keyword BM25 using SPLADE model
    hybrid → dense + sparse RRF fusion (best quality, recommended)

Embedding strategy:
    Uses metadata['embed_content'] for embedding when available.
    embed_content = "Title: {filename} | Section: {section}\\n{content}"
    This produces richer vectors with document context.
    Original page_content is preserved for LLM consumption.

Payload schema per chunk:
    doc_id, user_id, source, page, chunk_index,
    total_chunks, char_count, ingested_at

Async — all public methods are async def (Rule 1).
Sync internals run via asyncio.to_thread() (Rule 1).
"""

import asyncio
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
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from config.settings import settings
from utils.helpers import hash_text
from utils.logger import get_logger
from vectorstore.embeddings import get_embedding_dimension, get_embeddings
from vectorstore.base_store import BaseVectorStore

logger = get_logger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid"]


class QdrantStore(BaseVectorStore):
    """Qdrant vector store with dense, sparse, and hybrid search.

    Attributes:
        collection_name: Qdrant collection name.
        in_memory: If True, uses in-memory Qdrant (dev/testing).
        search_mode: Active search mode (dense/sparse/hybrid).
        _client: Single QdrantClient instance (shared, never duplicated).
        _store: LangChain QdrantVectorStore wrapper.
        _sparse_embeddings_instance: Lazy-loaded SPLADE sparse model.
    """

    # Sparse and dense vector config names in Qdrant collection
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
        """Sync constructor — stores config, no connections made.

        Args:
            collection_name: Qdrant collection name. Defaults to settings.
            in_memory: If True, use in-memory Qdrant (no server needed).
            qdrant_url: Qdrant server URL (overrides settings).
            qdrant_api_key: Qdrant API key (overrides settings).
            search_mode: Search strategy — 'dense', 'sparse', or 'hybrid'.
            client: Optional existing QdrantClient to reuse. When provided,
                skips client construction entirely — the caller's client is
                used as-is. Required so multiple collections share one
                in-memory database (same as connecting to one server in prod).
        """
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.in_memory = in_memory
        self.search_mode = search_mode
        self._qdrant_url = qdrant_url
        self._qdrant_api_key = qdrant_api_key
        self._injected_client: Optional[QdrantClient] = client

        # Initialized in initialize() — not here (two-phase pattern)
        self._client: Optional[QdrantClient] = None
        self._store: Optional[QdrantVectorStore] = None
        self._sparse_embeddings_instance: Optional[FastEmbedSparse] = None

    async def initialize(self) -> None:
        """Create client, collection, and vector store.

        Must be called once before add_documents() or similarity_search().
        Safe to call multiple times (idempotent).

        Raises:
            Exception: If connection or collection creation fails.
        """
        try:
            self._client = self._build_client()

            # Sync collection check runs in thread to avoid blocking
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
        """Build QdrantClient based on connection mode.

        If a client was injected at construction time, returns it directly
        (shared client — no new connection created).
        in_memory=True → QdrantClient(":memory:") for dev/testing.
        in_memory=False → connects to URL from settings or constructor.

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

        logger.info("QdrantStore: mode=server, url=%s", url)
        return QdrantClient(**kwargs)

    # Collection management

    def _create_collection_if_missing(self) -> None:
        """Create Qdrant collection with vector config for the search mode.

        Sync — runs inside asyncio.to_thread().

        Vector config by mode:
            dense  → only dense vectors (BGE cosine)
            sparse → only sparse vectors (SPLADE)
            hybrid → both dense + sparse (RRF fusion)

        Idempotent — skips if collection already exists.
        Logs a warning if existing collection may have mismatched config.
        """
        try:
            existing = [c.name for c in self._client.get_collections().collections]

            if self.collection_name in existing:
                logger.info(
                    "Collection '%s' already exists — skipping creation",
                    self.collection_name,
                )
                # Warn about potential schema mismatch
                self._validate_collection_config()
                return

            # Build vector configurations based on search mode
            vectors_config = {}
            sparse_vectors_config = {}

            if self.search_mode in ("dense", "hybrid"):
                vectors_config[self._DENSE_VECTOR_NAME] = VectorParams(
                    size=get_embedding_dimension(),
                    distance=Distance.COSINE,
                )

            if self.search_mode in ("sparse", "hybrid"):
                sparse_vectors_config[self._SPARSE_VECTOR_NAME] = SparseVectorParams()

            # Create collection
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config or None,
                sparse_vectors_config=sparse_vectors_config or None,
            )

            logger.info(
                "Created collection '%s': search_mode=%s",
                self.collection_name,
                self.search_mode,
            )

            # Create keyword indexes for all filtered fields.
            # Required on Qdrant Cloud/server — in-memory skips enforcement.
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
        """Warn if existing collection config doesn't match current search mode.

        Sync — called from _create_collection_if_missing().

        This is a soft check — logs a warning but doesn't fail.
        Full migration (adding sparse vectors to an existing dense-only
        collection) is complex and deferred to a manual admin step.
        """
        try:
            info = self._client.get_collection(self.collection_name)

            # Check if dense vectors exist when we need them
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
            # Validation is best-effort — don't fail on check errors
            logger.debug(
                "Could not validate collection config for '%s'",
                self.collection_name,
            )

    # Vector store construction

    def _build_vector_store(self) -> QdrantVectorStore:
        """Build LangChain QdrantVectorStore for the selected search mode.

        Sync — no I/O, just object construction.

        Mode mapping:
            dense  → RetrievalMode.DENSE  (semantic similarity)
            sparse → RetrievalMode.SPARSE (keyword matching)
            hybrid → RetrievalMode.HYBRID (RRF fusion)

        vector_name and sparse_vector_name must exactly match what was
        used in _create_collection_if_missing. Mismatch causes silent
        read/write failures.

        Returns:
            Configured QdrantVectorStore instance.

        Raises:
            ValueError: If search_mode is invalid.
        """
        valid_modes = ("dense", "sparse", "hybrid")
        if self.search_mode not in valid_modes:
            raise ValueError(
                f"Invalid search_mode '{self.search_mode}'. Must be one of {valid_modes}."
            )

        try:
            # LangChain retrieval mode mapping
            mode_map = {
                "dense": RetrievalMode.DENSE,
                "sparse": RetrievalMode.SPARSE,
                "hybrid": RetrievalMode.HYBRID,
            }

            # Common kwargs for all modes
            store_kwargs = {
                "client": self._client,
                "collection_name": self.collection_name,
                "retrieval_mode": mode_map[self.search_mode],
            }

            # Dense mode needs embedding model and vector name
            if self.search_mode in ("dense", "hybrid"):
                store_kwargs["embedding"] = get_embeddings()
                store_kwargs["vector_name"] = self._DENSE_VECTOR_NAME

            # Sparse mode needs SPLADE model and vector name
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
        """Return sparse embedding model (lazy-loaded, cached on self).

        Only instantiated when sparse or hybrid mode is used.
        Survives across multiple _build_vector_store calls.

        Returns:
            FastEmbedSparse instance for SPLADE model.
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
                    **kwargs,
                )
                logger.info("Sparse embedding model initialized successfully")
            except Exception as e:
                logger.exception("Error initializing sparse embeddings: %s", e)
                raise

        return self._sparse_embeddings_instance

    # Write — add documents

    async def add_documents(self, documents: List[Document]) -> List[str]:
        """Embed and store documents in Qdrant.

        Embedding strategy:
            Uses metadata['embed_content'] if available — this contains
            "Title: {file} | Section: {heading}\\n{content}" for richer
            semantic vectors. Falls back to page_content if not set.

            Original page_content is preserved in metadata['original_content']
            so similarity_search() can restore it for LLM consumption.

        Args:
            documents: List of Documents from Chunker.

        Returns:
            List of assigned Qdrant point IDs.

        Raises:
            Exception: If embedding or storage fails.
        """
        if not documents:
            logger.warning("add_documents received empty list")
            return []

        try:
            # Order matters: enrich first (char_count uses original page_content),
            # then dedup, then swap to embed_content. Do NOT reorder these calls.
            enriched_docs = self._enrich_metadata(documents)

            # Dedup — skip chunks already stored in Qdrant, saves all embedding work
            new_docs, skipped = await self._filter_existing_documents(enriched_docs)

            if not new_docs:
                logger.info(
                    "All %d chunks already stored — skipping embedding entirely",
                    len(documents),
                )
                return []

            embed_docs = self._prepare_for_embedding(new_docs)

            # LangChain embeds page_content and stores everything
            ids = await asyncio.to_thread(self._store.add_documents, embed_docs)

            logger.info(
                "QdrantStore stored %d new chunks, %d duplicates skipped",
                len(ids), skipped,
            )
            return ids

        except Exception as e:
            logger.exception("Error in add_documents: %s", e)
            raise

    async def _filter_existing_documents(
        self,
        documents: List[Document],
    ) -> tuple[List[Document], int]:
        """Filter out chunks already stored in Qdrant by chunk_id hash.

        The Chunker stores chunk_id = hash_text(page_content) in every
        chunk's metadata. We use that hash to check Qdrant in one batch
        query — no per-chunk round trips, no wasted embedding work.

        Falls back to ingesting all documents if the check fails — never
        blocks ingestion on a dedup error.

        Args:
            documents: Enriched documents from _enrich_metadata().

        Returns:
            Tuple of (new_documents_only, skipped_count).
        """
        if not documents:
            return documents, 0

        # chunk_id is set by Chunker._enrich_metadata() via hash_text(content).
        # Fall back to hashing page_content directly if somehow missing
        # (e.g. documents ingested outside the standard Chunker pipeline).
        chunk_ids = [
            doc.metadata.get("chunk_id") or hash_text(doc.page_content)
            for doc in documents
        ]

        try:
            # One Qdrant scroll query — find all points whose chunk_id is in
            # our batch. MatchAny is an OR across all values — far cheaper
            # than N individual queries (one per chunk).
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

            # Build set of already-stored chunk_ids from the scroll result
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
            # Never block ingestion on a dedup failure — log and proceed with all
            logger.warning(
                "Dedup check failed, ingesting all %d chunks: %s",
                len(documents), exc,
            )
            return documents, 0

    def _prepare_for_embedding(self, documents: List[Document]) -> List[Document]:
        """Swap page_content with embed_content for richer embeddings.

        LangChain QdrantVectorStore embeds whatever is in page_content.
        The Chunker puts richer text in metadata['embed_content'] that
        includes title and section context. We swap it in here so the
        embedding captures that context.

        Original page_content is saved in metadata['original_content']
        so similarity_search() can restore it for clean LLM consumption.

        Sync — pure data transformation, no I/O.

        Args:
            documents: Enriched documents ready for storage.

        Returns:
            New Document list with embed_content as page_content.
        """
        embed_docs = []

        for doc in documents:
            embed_content = doc.metadata.get("embed_content", doc.page_content)

            # Preserve original clean text for LLM retrieval
            metadata = doc.metadata.copy()
            metadata["original_content"] = doc.page_content

            embed_docs.append(
                Document(page_content=embed_content, metadata=metadata)
            )

        return embed_docs

    def _enrich_metadata(self, documents: List[Document]) -> List[Document]:
        """Attach required payload fields to every document.

        Caller can pre-set doc_id and user_id — setdefault won't overwrite.
        char_count and ingested_at are always overwritten for accuracy.

        Sync — pure data transformation, no I/O.

        Args:
            documents: Raw documents from Chunker.

        Returns:
            Documents with enriched metadata.
        """
        total_chunks = len(documents)
        enriched_docs = []

        for i, doc in enumerate(documents):
            metadata = doc.metadata.copy()

            # Identity fields — will be real UUIDs after auth layer
            metadata.setdefault("doc_id", "")
            metadata.setdefault("user_id", "")

            # Document info
            metadata.setdefault("source", "unknown")
            metadata.setdefault("page", 0)

            # Chunk position
            metadata.setdefault("chunk_index", i)
            metadata.setdefault("total_chunks", total_chunks)

            # Computed fields — always overwritten for accuracy
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

        Returns Documents with clean page_content (original text, not
        embed_content prefix). Relevance scores are attached to
        metadata['relevance_score'] when score_threshold is provided.

        Score threshold guide (BGE + cosine, dense mode):
            >= 0.7 → very confident match
            >= 0.5 → good match (recommended default)
            >= 0.3 → loose match
             < 0.3 → likely irrelevant

        Args:
            query: Search text to embed and compare.
            k: Number of results to return.
            score_threshold: Minimum similarity score (0.0-1.0).
                             None = return all top-k unfiltered.
            filter_user_id: Filter to specific user's documents.
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

            # Restore original page_content for LLM consumption
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
        """Search with relevance score filtering.

        Scores are preserved in metadata['relevance_score'] so the
        RAG layer can use them for reranking or confidence decisions.

        Args:
            query: Search query text.
            k: Number of results.
            score_threshold: Minimum score to include.
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

        # Filter by threshold and attach scores to metadata
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
        """Restore original page_content after retrieval.

        During add_documents(), page_content was swapped with embed_content
        for richer embeddings. Here we swap back so the LLM receives
        clean text without the "Title: ... | Section: ..." prefix.

        If original_content is not in metadata (old data or external
        ingestion), page_content is left unchanged.

        Sync — pure data transformation, no I/O.

        Args:
            documents: Documents from Qdrant search.

        Returns:
            Documents with clean page_content for LLM use.
        """
        restored = []
        for doc in documents:
            original = doc.metadata.get("original_content")
            if original is not None:
                doc = Document(page_content=original, metadata=doc.metadata)
            restored.append(doc)

        return restored

    def _build_filter(self, user_id: Optional[str]) -> Optional[Filter]:
        """Build Qdrant payload filter for user isolation.

        Args:
            user_id: User ID to filter by. None = no filter.

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
        """Search returning top-k results WITH their stored embedding vectors.

        Uses the raw QdrantClient.search(with_vectors=True) instead of the
        LangChain wrapper, so each result carries its stored dense vector.
        The vector is attached to metadata['vector'] on each Document and
        used by ContextRanker._rank_mmr() for inter-chunk similarity —
        eliminating the need to re-embed chunks on every query (~2-4s saved).

        Same content and score behaviour as similarity_search():
            - page_content is restored to original (not embed_content prefix)
            - relevance_score is attached to metadata['relevance_score']

        Args:
            query: Search query text.
            k: Number of results to return.
            filter_user_id: Filter to specific user's documents.

        Returns:
            List of Documents with clean page_content, relevance_score,
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

            # Raw Qdrant search — with_vectors=True returns the stored dense
            # vector alongside each result. Uses query_points (Qdrant client
            # v1.7+ API; replaces the deprecated client.search()).
            response = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self.collection_name,
                query=query_vector,
                using=self._DENSE_VECTOR_NAME,
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

                # Attach retrieval score (cosine similarity, 0.0-1.0)
                metadata["relevance_score"] = round(float(point.score), 4)

                # Extract the dense vector from the named-vector dict
                raw_vec = point.vector
                if isinstance(raw_vec, dict):
                    metadata["vector"] = raw_vec.get(self._DENSE_VECTOR_NAME)
                else:
                    metadata["vector"] = raw_vec  # fallback for unnamed vectors

                docs.append(Document(page_content=page_content, metadata=metadata))

            # Restore original page_content (strips embed_content prefix)
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
        """Hybrid RRF search returning top-k results WITH dense embedding vectors.

        Uses Qdrant's native prefetch + RRF fusion in a single query call —
        dense and sparse searches run in parallel inside Qdrant, results are
        fused via Reciprocal Rank Fusion, and dense vectors are returned
        alongside payloads for MMR diversity scoring.

        Args:
            query: Search query text.
            k: Number of results to return.
            filter_user_id: Filter to specific user's documents.

        Returns:
            List of Documents with clean page_content, relevance_score,
            and dense embedding vector in metadata['vector'].

        Raises:
            Exception: If hybrid search fails.
        """
        if not query or not query.strip():
            logger.warning("hybrid_search_with_vectors received empty query")
            return []
        if k <= 0:
            return []

        try:
            # Dense query vector
            embeddings_model = get_embeddings()
            query_vector = await asyncio.to_thread(
                embeddings_model.embed_query, query
            )

            # Sparse query vector from SPLADE/BM25
            sparse_model = self._get_sparse_embeddings()
            sparse_vector = await asyncio.to_thread(
                sparse_model.embed_query, query
            )

            qdrant_filter = self._build_filter(filter_user_id)

            # Fetch more candidates for each leg so RRF has enough to fuse
            coarse_k = max(k * 3, 20)

            response = await asyncio.to_thread(
                self._client.query_points,
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=query_vector,
                        using=self._DENSE_VECTOR_NAME,
                        limit=coarse_k,
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

                # Extract dense vector for MMR inter-chunk diversity scoring
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

        Irreversible — all vectors and metadata are lost.
        Use with extreme caution.

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
        """Return collection statistics for observability.

        Returns:
            Dict with backend, collection, count, model, mode info.
            Returns empty dict on failure (never raises).
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
        """Graceful shutdown — close client connection and release resources.

        Prevents connection leaks. Safe to call multiple times.

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

        # Release references
        self._client = None
        self._store = None