"""
Unit tests for the RAG layer — no external services required.

Test scope:
    Unit tests (no pytest) covering RAGRequest/RAGConfig/RAGResponse models,
    the RAG exception hierarchy, prompt template builders, ContextRanker
    strategies, ContextAssembler token budgeting, SimpleRAG end-to-end pipeline,
    CorrectiveRAG pass/retry/low-confidence branches, and RAGFactory creation,
    registration, and validation.

Flow:
    Phases 1-8 run sequentially; each phase calls _report() for each assertion
    and exits with sys.exit(1) if any test fails.

Dependencies:
    MockLLM (canned responses, call-count tracking), MockRetriever (pre-built
    chunks), MockRelevanceEvalLLM (configurable relevance scores); no Qdrant,
    Redis, API keys, or embedding model.
"""

import asyncio
import sys
import time

from pydantic import ValidationError

from llm.models.llm_response import LLMResponse
from llm.contracts.base_llm import BaseLLM
from rag.models.rag_request import (
    RAGRequest,
    RAGConfig,
    ConversationTurn,
    MetadataFilter,
    SUPPORTED_RAG_VARIANTS,
    SUPPORTED_RETRIEVAL_MODES,
)
from rag.models.rag_response import (
    RAGResponse,
    RetrievedChunk,
    ConfidenceScore,
    RAGTimings,
)
from rag.exceptions.rag_exceptions import (
    RAGError,
    RAGConfigError,
    RAGRetrievalError,
    RAGContextError,
    RAGGenerationError,
    RAGQualityError,
)
from rag.retrieval.base_retriever import BaseRetriever
from rag.context.context_assembler import ContextAssembler
from rag.context.context_ranker import ContextRanker
from rag.prompts.rag_prompt_templates import (
    build_rag_prompt,
    build_relevance_eval_prompt,
    build_query_rewrite_prompt,
    format_conversation_history,
)
from rag.variants.simple_rag import SimpleRAG
from rag.variants.corrective_rag import CorrectiveRAG
from rag.rag_factory import RAGFactory
from rag.base_rag import BaseRAG
from utils.logger import get_logger

logger = get_logger(__name__)


_pass_count = 0
_fail_count = 0


def _report(test_name: str, passed: bool, detail: str = "") -> None:
    """Update pass/fail counters and print a labelled result line."""
    global _pass_count, _fail_count
    if passed:
        _pass_count += 1
        print(f"  [PASS] {test_name} {detail}")
    else:
        _fail_count += 1
        print(f"  [FAIL] {test_name} {detail}")
        logger.error("[FAIL] %s %s", test_name, detail)


def separator(label: str) -> None:
    """Print a visual phase separator."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


def _make_chunk(
    content: str = "Test chunk content",
    source: str = "test.pdf",
    score: float = 0.85,
    chunk_id: str = "abc123",
) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for use in tests."""
    return RetrievedChunk(
        content=content,
        source_file=source,
        chunk_id=chunk_id,
        relevance_score=score,
    )


def _make_chunks(count: int = 5) -> list[RetrievedChunk]:
    """Build a list of test chunks with decreasing relevance scores."""
    chunks = []
    for i in range(count):
        chunks.append(_make_chunk(
            content=f"Chunk {i+1} content about topic {chr(65+i)}.",
            source=f"doc_{i+1}.pdf",
            score=round(0.95 - i * 0.05, 2),
            chunk_id=f"chunk_{i+1}",
        ))
    return chunks


class MockLLM(BaseLLM):
    """Mock LLM returning configurable canned responses without API calls.

    Tracks call_count so tests can assert how many times the LLM was invoked.
    """

    def __init__(
        self,
        response_text: str = "This is a mock LLM response.",
        token_count: int = 10,
    ) -> None:
        self._response_text = response_text
        self._token_count = token_count
        self.call_count = 0
        self.last_messages = None

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-model-v1"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            text=self._response_text,
            model=self.model_name,
            provider="gemini",
            finish_reason="stop",
            prompt_tokens=len(prompt.split()),
            completion_tokens=5,
            tokens_used=len(prompt.split()) + 5,
            latency_ms=10.0,
        )

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        if not messages:
            raise ValueError("Messages list cannot be empty.")
        self.call_count += 1
        self.last_messages = messages
        content = messages[-1].get("content", "")
        return LLMResponse(
            text=self._response_text,
            model=self.model_name,
            provider="gemini",
            finish_reason="stop",
            prompt_tokens=len(content.split()),
            completion_tokens=5,
            tokens_used=len(content.split()) + 5,
            latency_ms=10.0,
        )

    async def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return self._token_count

    async def is_available(self) -> bool:
        return True


class MockRetriever(BaseRetriever):
    """Mock retriever that returns pre-built chunks."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        super().__init__(store=None)
        self._chunks = chunks or _make_chunks()
        self.call_count = 0
        self.last_query = None

    @property
    def retriever_type(self) -> str:
        return "mock"

    async def retrieve(self, query, top_k=5, filters=None):
        self.call_count += 1
        self.last_query = query
        return self._chunks[:top_k]


class MockRelevanceEvalLLM(BaseLLM):
    """Mock LLM for CorrectiveRAG that returns configurable relevance scores.

    Distinguishes eval, rewrite, and generation calls by inspecting system
    message content, and increments per-type counters for assertion.
    """

    def __init__(
        self,
        relevance_score: float = 0.85,
        rewrite_text: str = "rewritten query about the topic",
        generation_text: str = "Generated answer from context.",
    ) -> None:
        self._relevance_score = relevance_score
        self._rewrite_text = rewrite_text
        self._generation_text = generation_text
        self.eval_count = 0
        self.rewrite_count = 0
        self.generation_count = 0
        self.total_calls = 0

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-eval-v1"

    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        self.total_calls += 1
        return self._build_response(self._generation_text, len(prompt.split()))

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        self.total_calls += 1
        last_content = messages[-1].get("content", "")
        system_content = messages[0].get("content", "") if messages else ""

        # Route by system message keywords
        if "relevance" in system_content.lower() and "evaluator" in system_content.lower():
            self.eval_count += 1
            text = f'{{"relevance": {self._relevance_score}, "reason": "test eval"}}'
            return self._build_response(text, len(last_content.split()))

        if "rewrite" in system_content.lower():
            self.rewrite_count += 1
            return self._build_response(self._rewrite_text, len(last_content.split()))

        self.generation_count += 1
        return self._build_response(self._generation_text, len(last_content.split()))

    async def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text.split()))

    async def is_available(self) -> bool:
        return True

    def _build_response(self, text: str, prompt_tokens: int) -> LLMResponse:
        return LLMResponse(
            text=text,
            model=self.model_name,
            provider="gemini",
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=5,
            tokens_used=prompt_tokens + 5,
            latency_ms=10.0,
        )


def test_models():
    """Verifies RAGRequest, RAGConfig, RAGResponse, and supporting model validation rules.

    Tests:
        - Valid RAGRequest construction; empty and whitespace queries are rejected.
        - RAGConfig defaults; invalid variant, zero/out-of-bound top_k, and negative temperature raise.
        - ConversationTurn is frozen and rejects empty content.
        - MetadataFilter rejects unsupported operators.
        - RetrievedChunk is frozen and converts from a document via from_document().
        - RAGResponse.from_generation sets expected fields; from_cache sets cache_hit=True.
        - cache_layer without cache_hit=True is rejected.
        - get_chat_messages() returns message dicts or None when no history present.
    """
    separator("Phase 1: Model Validation")

    # Valid construction

    req = RAGRequest(query="What is RAG?", collection_name="docs")
    _report(
        "request_valid",
        req.query == "What is RAG?"
        and req.collection_name == "docs"
        and req.config is not None
        and len(req.request_id) > 0,
    )

    try:
        RAGRequest(query="", collection_name="docs")
        _report("request_empty_query", False)
    except (ValidationError, ValueError):
        _report("request_empty_query", True)

    try:
        RAGRequest(query="   \n", collection_name="docs")
        _report("request_whitespace_query", False)
    except (ValidationError, ValueError):
        _report("request_whitespace_query", True)

    cfg = RAGConfig()
    _report(
        "config_defaults",
        cfg.rag_variant is None
        and cfg.retrieval_mode == "dense"
        and cfg.top_k == 5
        and cfg.rerank_strategy == "mmr"
        and cfg.temperature == 0.3
        and cfg.include_sources is True
        and cfg.confidence_method == "retrieval",
    )

    try:
        RAGConfig(rag_variant="nonexistent")
        _report("config_invalid_variant", False)
    except (ValidationError, ValueError):
        _report("config_invalid_variant", True)

    cfg2 = RAGConfig(rag_variant="corrective")
    _report("config_valid_variant", cfg2.rag_variant == "corrective")

    cfg3 = RAGConfig()
    resolved = cfg3.resolve_variant()
    _report("config_resolve_default", resolved in SUPPORTED_RAG_VARIANTS or resolved == "simple")

    cfg4 = RAGConfig(rag_variant="corrective")
    _report("config_resolve_explicit", cfg4.resolve_variant() == "corrective")

    try:
        RAGConfig(top_k=0)
        _report("config_top_k_zero", False)
    except (ValidationError, ValueError):
        _report("config_top_k_zero", True)

    try:
        RAGConfig(top_k=51)
        _report("config_top_k_over", False)
    except (ValidationError, ValueError):
        _report("config_top_k_over", True)

    try:
        RAGConfig(temperature=-0.1)
        _report("config_temp_negative", False)
    except (ValidationError, ValueError):
        _report("config_temp_negative", True)

    turn = ConversationTurn(role="user", content="Hello")
    _report("conv_turn_valid", turn.role == "user" and turn.content == "Hello")

    try:
        turn.role = "assistant"
        _report("conv_turn_frozen", False)
    except (ValidationError, AttributeError, TypeError):
        _report("conv_turn_frozen", True)

    try:
        ConversationTurn(role="user", content="  ")
        _report("conv_turn_empty", False)
    except (ValidationError, ValueError):
        _report("conv_turn_empty", True)

    mf = MetadataFilter(field="source", value="test.pdf")
    _report("meta_filter_valid", mf.field == "source" and mf.operator == "eq")

    try:
        MetadataFilter(field="source", value="x", operator="regex")
        _report("meta_filter_bad_op", False)
    except (ValidationError, ValueError):
        _report("meta_filter_bad_op", True)

    chunk = _make_chunk()
    _report("chunk_valid", chunk.content == "Test chunk content" and chunk.used_in_context is False)

    try:
        chunk.used_in_context = True
        _report("chunk_frozen", False)
    except (ValidationError, AttributeError, TypeError):
        _report("chunk_frozen", True)

    class FakeDoc:
        page_content = "Document text"
        metadata = {"source_file": "report.pdf", "page_number": 3, "extra_field": "value"}

    converted = RetrievedChunk.from_document(FakeDoc(), relevance_score=0.9)
    _report(
        "chunk_from_document",
        converted.content == "Document text"
        and converted.source_file == "report.pdf"
        and converted.page_number == 3
        and converted.relevance_score == 0.9
        and "extra_field" in converted.metadata,
    )

    llm_resp = LLMResponse(
        text="Test answer", model="test", provider="gemini",
        finish_reason="stop", prompt_tokens=10, completion_tokens=5, tokens_used=15,
    )
    rag_resp = RAGResponse.from_generation(
        answer="Test answer",
        llm_response=llm_resp,
        sources=[_make_chunk()],
        timings=RAGTimings(retrieval_ms=10, generation_ms=50, total_ms=60),
        confidence=ConfidenceScore(value=0.85, method="retrieval"),
        request_id="test-123",
        rag_variant="simple",
    )
    _report(
        "response_from_generation",
        rag_resp.answer == "Test answer"
        and rag_resp.cache_hit is False
        and rag_resp.rag_variant == "simple"
        and rag_resp.confidence.value == 0.85
        and len(rag_resp.sources) == 1,
    )

    cached_resp = RAGResponse.from_cache(
        cached_response=llm_resp,
        request_id="test-456",
        rag_variant="simple",
        cache_layer="L1",
    )
    _report(
        "response_from_cache",
        cached_resp.cache_hit is True
        and cached_resp.cache_layer == "L1"
        and cached_resp.confidence.method == "cache",
    )

    try:
        RAGResponse(
            answer="test", confidence=ConfidenceScore(value=0.5, method="test"),
            rag_variant="simple", cache_hit=False, cache_layer="L1",
        )
        _report("response_cache_consistency", False)
    except (ValidationError, ValueError):
        _report("response_cache_consistency", True)

    req_with_history = RAGRequest(
        query="Tell me more",
        collection_name="docs",
        conversation_history=[
            ConversationTurn(role="user", content="What is RAG?"),
            ConversationTurn(role="assistant", content="RAG is..."),
        ],
    )
    msgs = req_with_history.get_chat_messages()
    _report(
        "request_chat_messages",
        msgs is not None
        and len(msgs) == 2
        and msgs[0]["role"] == "user",
    )

    req_no_hist = RAGRequest(query="Hello", collection_name="docs")
    _report("request_no_history", req_no_hist.get_chat_messages() is None)


def test_exceptions():
    """Verifies the RAG exception hierarchy and details dict propagation.

    Tests:
        - All specific error classes inherit from RAGError.
        - RAGError inherits from Exception.
        - details dict is preserved on the exception instance.
        - All subclasses are caught by a bare except RAGError handler.
    """
    separator("Phase 2: Exception Hierarchy")

    subclasses = [RAGConfigError, RAGRetrievalError, RAGContextError,
                  RAGGenerationError, RAGQualityError]

    all_inherit = all(issubclass(c, RAGError) for c in subclasses)
    _report("all_inherit_from_rag_error", all_inherit)

    _report("rag_error_is_exception", issubclass(RAGError, Exception))

    exc = RAGRetrievalError("retrieval failed", details={"top_k": 5})
    _report("exception_details", exc.details == {"top_k": 5})

    all_caught = True
    for cls in subclasses:
        try:
            raise cls("test")
        except RAGError:
            pass
        except Exception:
            all_caught = False
    _report("exception_catch_all", all_caught)


def test_prompts():
    """Verifies that prompt builder functions produce correctly formatted output.

    Tests:
        - build_rag_prompt includes the query and context; adds history section when present.
        - build_relevance_eval_prompt includes the query and document.
        - build_query_rewrite_prompt includes the query.
        - format_conversation_history renders role/content pairs; returns empty string for [].
        - max_turns limit discards older turns and retains recent ones.
    """
    separator("Phase 3: Prompt Templates")

    sys, user = build_rag_prompt(query="What is RAG?", context="RAG is a technique.")
    _report(
        "rag_prompt_basic",
        "ONLY" in sys
        and "What is RAG?" in user
        and "RAG is a technique." in user,
    )

    sys2, user2 = build_rag_prompt(
        query="Tell me more",
        context="More details here.",
        conversation_history="User: What is RAG?\nAssistant: RAG is...",
    )
    _report("rag_prompt_with_history", "Conversation so far" in user2)

    sys3, user3 = build_relevance_eval_prompt(
        query="What is RAG?",
        document="RAG combines retrieval with generation.",
    )
    _report(
        "relevance_eval_prompt",
        "relevance" in sys3.lower()
        and "What is RAG?" in user3
        and "RAG combines" in user3,
    )

    sys4, user4 = build_query_rewrite_prompt("What is RAG?")
    _report("query_rewrite_prompt", "rewrite" in sys4.lower() and "What is RAG?" in user4)

    turns = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    formatted = format_conversation_history(turns)
    _report("format_history", "User: Hello" in formatted and "Assistant: Hi there" in formatted)

    _report("format_history_empty", format_conversation_history([]) == "")

    many_turns = [{"role": "user", "content": f"Turn {i}"} for i in range(20)]
    limited = format_conversation_history(many_turns, max_turns=5)
    _report("format_history_limited", "Turn 15" in limited and "Turn 0" not in limited)


def test_ranker():
    """Verifies ContextRanker strategy selection, fallback behaviour, and per-call overrides.

    Tests:
        - 'none' strategy preserves input order.
        - Empty and single-chunk inputs are handled without error.
        - 'mmr' without an embeddings function falls back to 'none'.
        - 'cross_encoder' and unknown strategies fall back to 'none'.
        - mmr_lambda outside [0, 1] raises ValueError.
        - A per-call strategy override takes precedence over the instance default.
    """
    separator("Phase 4: Context Ranker")

    chunks = _make_chunks(5)

    ranker_none = ContextRanker(strategy="none")
    result_none = asyncio.run(ranker_none.rank(chunks, "test query"))
    _report("ranker_none_preserves_order", result_none == chunks)

    result_empty = asyncio.run(ranker_none.rank([], "test"))
    _report("ranker_empty_input", result_empty == [])

    result_single = asyncio.run(ranker_none.rank([chunks[0]], "test"))
    _report("ranker_single_chunk", len(result_single) == 1)

    # MMR falls back to 'none' when no embeddings function is provided
    ranker_mmr = ContextRanker(strategy="mmr", embeddings_fn=None)
    result_mmr = asyncio.run(ranker_mmr.rank(chunks, "test query"))
    _report("ranker_mmr_fallback", len(result_mmr) == 5)

    ranker_ce = ContextRanker(strategy="cross_encoder")
    result_ce = asyncio.run(ranker_ce.rank(chunks, "test query"))
    _report("ranker_cross_encoder_fallback", len(result_ce) == 5)

    ranker_unk = ContextRanker(strategy="unknown_strategy")
    result_unk = asyncio.run(ranker_unk.rank(chunks, "test"))
    _report("ranker_unknown_fallback", len(result_unk) == 5)

    try:
        ContextRanker(mmr_lambda=1.5)
        _report("ranker_invalid_lambda", False)
    except ValueError:
        _report("ranker_invalid_lambda", True)

    ranker_default_mmr = ContextRanker(strategy="mmr")
    result_override = asyncio.run(ranker_default_mmr.rank(chunks, "test", strategy="none"))
    _report("ranker_per_call_override", result_override == chunks)


def test_assembler():
    """Verifies ContextAssembler token budgeting, source labels, and error handling.

    Tests:
        - Normal assembly produces a non-empty context string with positive token count.
        - used_in_context flags are set on at least one chunk.
        - [Source N:] labels appear in the output by default.
        - A very tight token budget restricts the number of included chunks.
        - Assembling an empty list raises RAGContextError.
        - max_tokens below the minimum raises ValueError at construction time.
        - include_source_labels=False omits the [Source] prefix.
    """
    separator("Phase 5: Context Assembler")

    llm = MockLLM(token_count=10)
    chunks = _make_chunks(5)

    assembler = ContextAssembler(llm=llm, max_tokens=500)
    ctx_str, updated, tokens = asyncio.run(assembler.assemble(chunks))
    _report("assembler_basic", len(ctx_str) > 0 and tokens > 0)

    used_count = sum(1 for c in updated if c.used_in_context)
    _report("assembler_used_in_context", used_count > 0)

    _report("assembler_source_labels", "[Source 1:" in ctx_str)

    tight_assembler = ContextAssembler(llm=llm, max_tokens=25)
    ctx_tight, updated_tight, tokens_tight = asyncio.run(
        tight_assembler.assemble(chunks)
    )
    used_tight = sum(1 for c in updated_tight if c.used_in_context)
    _report("assembler_budget_limits", used_tight < len(chunks))

    try:
        asyncio.run(assembler.assemble([]))
        _report("assembler_empty_raises", False)
    except RAGContextError:
        _report("assembler_empty_raises", True)

    try:
        ContextAssembler(llm=llm, max_tokens=10)
        _report("assembler_min_tokens", False)
    except ValueError:
        _report("assembler_min_tokens", True)

    no_label_asm = ContextAssembler(llm=llm, max_tokens=500, include_source_labels=False)
    ctx_no_label, _, _ = asyncio.run(no_label_asm.assemble(chunks))
    _report("assembler_no_labels", "[Source" not in ctx_no_label)


def test_simple_rag():
    """Verifies SimpleRAG end-to-end retrieval→generation pipeline with mock infrastructure.

    Tests:
        - variant_name is 'simple'; repr contains class and provider name.
        - Full pipeline returns a RAGResponse with non-empty answer, correct variant,
          matching request_id, cache_hit=False, sources, confidence, timings, and model_name.
        - Retriever and LLM are each called at least once.
        - include_sources=False omits sources from the response.
        - used_in_context is set on at least one source chunk.
    """
    separator("Phase 6: SimpleRAG Pipeline")

    llm = MockLLM(response_text="RAG is a technique for grounded generation.", token_count=10)
    retriever = MockRetriever()
    rag = SimpleRAG(retriever=retriever, llm=llm)

    _report("simple_variant_name", rag.variant_name == "simple")

    repr_str = repr(rag)
    _report("simple_repr", "SimpleRAG" in repr_str and "mock" in repr_str)

    request = RAGRequest(query="What is RAG?", collection_name="docs")
    response = asyncio.run(rag.query(request))

    _report("simple_returns_rag_response", isinstance(response, RAGResponse))
    _report("simple_answer_not_empty", len(response.answer) > 0)
    _report("simple_variant_in_response", response.rag_variant == "simple")
    _report("simple_request_id_flows", response.request_id == request.request_id)
    _report("simple_cache_hit_false", response.cache_hit is False)
    _report("simple_sources_present", len(response.sources) > 0)
    _report("simple_confidence_present", response.confidence.value > 0)
    _report("simple_timings_present", response.timings.total_ms > 0)
    _report("simple_model_name", response.model_name == "mock-model-v1")
    _report("simple_low_confidence_false", response.low_confidence is False)

    _report("simple_retriever_called", retriever.call_count == 1)
    _report("simple_llm_called", llm.call_count >= 1)

    request2 = RAGRequest(
        query="Explain RAG",
        collection_name="docs",
        config=RAGConfig(top_k=3, temperature=0.1, include_sources=False),
    )
    response2 = asyncio.run(rag.query(request2))
    _report("simple_no_sources", len(response2.sources) == 0)

    request3 = RAGRequest(query="Test", collection_name="docs")
    response3 = asyncio.run(rag.query(request3))
    used = [s for s in response3.sources if s.used_in_context]
    _report("simple_used_in_context_set", len(used) > 0)


def test_corrective_rag():
    """Verifies CorrectiveRAG PASS, RETRY, and LOW_CONFIDENCE branch behaviour.

    Tests:
        - PASS branch (relevance=0.85): no rewrite, retriever called once, low_confidence=False.
        - RETRY branch (relevance=0.50): rewrite attempted, retriever called at least twice.
        - LOW_CONFIDENCE branch (relevance=0.15): low_confidence=True, no rewrite attempted.
        - _parse_relevance_score handles JSON, markdown-fenced JSON, raw float, clamping, and None.
    """
    separator("Phase 7: CorrectiveRAG")

    high_llm = MockRelevanceEvalLLM(relevance_score=0.85)
    retriever = MockRetriever()
    crag_pass = CorrectiveRAG(
        retriever=retriever,
        llm=high_llm,
        pass_threshold=0.7,
        retry_threshold=0.4,
        max_retries=1,
    )

    _report("crag_variant_name", crag_pass.variant_name == "corrective")

    req = RAGRequest(query="What is HIPAA?", collection_name="legal")
    resp_pass = asyncio.run(crag_pass.query(req))

    _report("crag_pass_answer", len(resp_pass.answer) > 0)
    _report("crag_pass_low_conf_false", resp_pass.low_confidence is False)
    _report("crag_pass_variant", resp_pass.rag_variant == "corrective")
    _report("crag_pass_confidence", resp_pass.confidence.value > 0.7)
    _report("crag_pass_confidence_method", resp_pass.confidence.method == "corrective_eval")
    _report("crag_pass_retriever_once", retriever.call_count == 1)
    _report(
        "crag_pass_eval_count",
        high_llm.eval_count == 3,
        f"| expected 3, got {high_llm.eval_count}",
    )
    _report("crag_pass_no_rewrite", high_llm.rewrite_count == 0)

    med_llm = MockRelevanceEvalLLM(relevance_score=0.50)
    retriever2 = MockRetriever()

    # Mock stays at 0.50 throughout; we verify that a rewrite was attempted
    crag_retry = CorrectiveRAG(
        retriever=retriever2,
        llm=med_llm,
        pass_threshold=0.7,
        retry_threshold=0.4,
        max_retries=1,
    )

    req2 = RAGRequest(query="HIPAA penalties", collection_name="legal")
    resp_retry = asyncio.run(crag_retry.query(req2))

    _report("crag_retry_answer", len(resp_retry.answer) > 0)
    _report(
        "crag_retry_rewrite_attempted",
        med_llm.rewrite_count >= 1,
        f"| rewrite_count={med_llm.rewrite_count}",
    )
    _report(
        "crag_retry_retriever_called_twice",
        retriever2.call_count >= 2,
        f"| call_count={retriever2.call_count}",
    )
    _report(
        "crag_retry_evals_doubled",
        med_llm.eval_count >= 6,
        f"| eval_count={med_llm.eval_count}",
    )

    low_llm = MockRelevanceEvalLLM(relevance_score=0.15)
    retriever3 = MockRetriever()

    crag_low = CorrectiveRAG(
        retriever=retriever3,
        llm=low_llm,
        pass_threshold=0.7,
        retry_threshold=0.4,
        max_retries=1,
    )

    req3 = RAGRequest(query="Quantum physics in law", collection_name="legal")
    resp_low = asyncio.run(crag_low.query(req3))

    _report("crag_low_answer", len(resp_low.answer) > 0)
    _report("crag_low_confidence_true", resp_low.low_confidence is True)
    _report("crag_low_no_rewrite", low_llm.rewrite_count == 0)
    _report(
        "crag_low_retriever_once",
        retriever3.call_count == 1,
        f"| call_count={retriever3.call_count}",
    )

    crag_test = CorrectiveRAG(
        retriever=MockRetriever(),
        llm=MockLLM(),
    )

    score_json = crag_test._parse_relevance_score('{"relevance": 0.85, "reason": "good"}')
    _report("parse_score_json", score_json == 0.85)

    score_fenced = crag_test._parse_relevance_score('```json\n{"relevance": 0.72}\n```')
    _report("parse_score_fenced", score_fenced == 0.72)

    score_raw = crag_test._parse_relevance_score("The relevance is 0.65")
    _report("parse_score_fallback", score_raw == 0.65)

    score_over = crag_test._parse_relevance_score('{"relevance": 1.5}')
    _report("parse_score_clamp", score_over == 1.0)

    score_none = crag_test._parse_relevance_score("no numbers here")
    _report("parse_score_none", score_none is None)

    score_empty = crag_test._parse_relevance_score("")
    _report("parse_score_empty", score_empty is None)


def test_factory():
    """Verifies RAGFactory variant creation, registration, and input validation.

    Tests:
        - available_variants() and available_retrieval_modes() include expected keys.
        - create() returns SimpleRAG or CorrectiveRAG for matching variant names.
        - Variant names are case-insensitive.
        - Unknown or empty variant names raise RAGConfigError.
        - create_from_request() selects the correct class from config.rag_variant.
        - create_retriever() constructs a dense retriever; unknown mode raises RAGConfigError.
        - register_variant() adds and can later be cleaned up; non-BaseRAG class raises RAGConfigError.
    """
    separator("Phase 8: RAGFactory")

    llm = MockLLM()
    retriever = MockRetriever()

    variants = RAGFactory.available_variants()
    _report("factory_variants", "simple" in variants and "corrective" in variants)

    modes = RAGFactory.available_retrieval_modes()
    _report("factory_modes", "dense" in modes and "hybrid" in modes)

    simple = RAGFactory.create("simple", retriever=retriever, llm=llm)
    _report("factory_create_simple", isinstance(simple, SimpleRAG))

    corrective = RAGFactory.create("corrective", retriever=retriever, llm=llm)
    _report("factory_create_corrective", isinstance(corrective, CorrectiveRAG))

    upper = RAGFactory.create("SIMPLE", retriever=retriever, llm=llm)
    _report("factory_case_insensitive", isinstance(upper, SimpleRAG))

    try:
        RAGFactory.create("nonexistent", retriever=retriever, llm=llm)
        _report("factory_unknown_variant", False)
    except RAGConfigError:
        _report("factory_unknown_variant", True)

    try:
        RAGFactory.create("", retriever=retriever, llm=llm)
        _report("factory_empty_variant", False)
    except RAGConfigError:
        _report("factory_empty_variant", True)

    req_simple = RAGRequest(
        query="test",
        collection_name="docs",
        config=RAGConfig(rag_variant="simple"),
    )
    from_req = RAGFactory.create_from_request(
        request=req_simple,
        store=None,
        llm=llm,
    )
    _report("factory_from_request_simple", isinstance(from_req, SimpleRAG))

    req_corrective = RAGRequest(
        query="test",
        collection_name="docs",
        config=RAGConfig(rag_variant="corrective"),
    )
    from_req2 = RAGFactory.create_from_request(
        request=req_corrective,
        store=None,
        llm=llm,
    )
    _report("factory_from_request_corrective", isinstance(from_req2, CorrectiveRAG))

    req_default = RAGRequest(query="test", collection_name="docs")
    from_req3 = RAGFactory.create_from_request(
        request=req_default,
        store=None,
        llm=llm,
    )
    _report("factory_from_request_default", isinstance(from_req3, BaseRAG))

    dense = RAGFactory.create_retriever(store=None, mode="dense")
    _report("factory_create_dense_retriever", dense.retriever_type == "dense")

    try:
        RAGFactory.create_retriever(store=None, mode="quantum")
        _report("factory_unknown_mode", False)
    except RAGConfigError:
        _report("factory_unknown_mode", True)

    class _TestRAG(BaseRAG):
        @property
        def variant_name(self):
            return "test_variant"

        async def retrieve(self, query, top_k, filters=None):
            return []

    try:
        RAGFactory.register_variant("_test_variant", _TestRAG)
        _report("factory_register_valid", "_test_variant" in RAGFactory.available_variants())
    finally:
        RAGFactory._variant_registry.pop("_test_variant", None)

    # register_variant — invalid class
    try:
        RAGFactory.register_variant("bad", str)
        _report("factory_register_invalid", False)
    except RAGConfigError:
        _report("factory_register_invalid", True)


# # My CUstom test case
#     try:
#         print(f"="*40)
#         logger.info(f"MY CUSTOM TEST CASE.")
#         print(f"="*40)
#         result = asyncio.run(
#             RAGFactory.create("simple", retriever=retriever, llm=llm).query(
#                 RAGRequest(query="Who is cristiano Ronaldo?", collection_name="football")
#             )
#         )
#         print(f"="*40)
#         logger.info(result)
#         print(f"="*40)
#         _report("custom_test_case", result is not None)
#     except Exception as e:
#         _report("custom_test_case", False)
#         logger.error("Custom test case failed: %s", e)

def run_all_tests():
    """Run all test phases sequentially and exit with a non-zero code on any failure."""
    global _pass_count, _fail_count
    _pass_count = 0
    _fail_count = 0

    start = time.monotonic()

    separator("RAG LAYER TEST SUITE")

    test_models()
    test_exceptions()
    test_prompts()
    test_ranker()
    test_assembler()
    test_simple_rag()
    test_corrective_rag()
    test_factory()

    elapsed = (time.monotonic() - start) * 1000

    separator("TEST SUMMARY")
    print(f"  Passed  : {_pass_count}")
    print(f"  Failed  : {_fail_count}")
    print(f"  Total   : {_pass_count + _fail_count}")
    print(f"  Time    : {elapsed:.0f} ms")
    print()

    if _fail_count > 0:
        logger.error(
            "RAG TEST SUITE FAILED | passed=%d | failed=%d | elapsed=%.0f ms",
            _pass_count, _fail_count, elapsed,
        )
        sys.exit(1)
    else:
        logger.info(
            "RAG TEST SUITE PASSED | passed=%d | failed=%d | elapsed=%.0f ms",
            _pass_count, _fail_count, elapsed,
        )
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()