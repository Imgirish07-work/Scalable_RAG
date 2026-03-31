"""
Real agent layer tests — no pytest, runs with: python test_agents.py

Tests every component with realistic mock dependencies:
    - MockLLM:       canned JSON responses for planning, verification, synthesis
    - MockPipeline:  returns pre-built RAGResponse for sub-query execution

Test sections:
    1. ComplexityDetector  — should_decompose() heuristics
    2. QueryPlanner        — LLM call + JSON parsing + fallback
    3. ParallelRetriever   — concurrent + sequential execution
    4. ResultVerifier      — heuristic checks (length, confidence, non-answers)
    5. AnswerSynthesizer   — synthesis call + empty-output guard
    6. AgentOrchestrator   — full flow end-to-end with mocks
    7. Pipeline integration — configure_agents() hook on RAGPipeline
"""

import asyncio
import json
import traceback

from agents.agent_orchestrator import AgentOrchestrator
from agents.exceptions.agent_exceptions import (
    AgentPlanningError,
    AgentRetrievalError,
    AgentSynthesisError,
)
from agents.models.agent_request import DecompositionPlan, SubQuery
from agents.models.agent_response import AgentResponse, SubQueryResult
from agents.planner.complexity_detector import should_decompose
from agents.planner.query_planner import QueryPlanner
from agents.retriever.parallel_retriever import ParallelRetriever
from agents.synthesizer.answer_synthesizer import AnswerSynthesizer
from agents.verifier.result_verifier import ResultVerifier
from llm.models.llm_response import LLMResponse
from rag.models.rag_request import RAGConfig, RAGRequest
from rag.models.rag_response import ConfidenceScore, RAGResponse, RAGTimings, RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Test result tracking
# ──────────────────────────────────────────────────────────────────────────────

_passed = 0
_failed = 0
_errors = []


def _ok(label: str) -> None:
    global _passed
    _passed += 1
    print(f"  [PASS] {label}")


def _fail(label: str, reason: str) -> None:
    global _failed
    _failed += 1
    _errors.append(f"{label}: {reason}")
    print(f"  [FAIL] {label} — {reason}")


def check(label: str, condition: bool, reason: str = "") -> None:
    if condition:
        _ok(label)
    else:
        _fail(label, reason or "condition was False")


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ──────────────────────────────────────────────────────────────────────────────
# Mock dependencies
# ──────────────────────────────────────────────────────────────────────────────

class MockLLM:
    """Minimal LLM mock. Returns canned responses keyed by scenario."""

    provider_name = "mock"
    model_name = "mock-model"

    def __init__(self, chat_responses: list[str]) -> None:
        self._responses = list(chat_responses)
        self._call_count = 0

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        if self._call_count < len(self._responses):
            text = self._responses[self._call_count]
        else:
            text = self._responses[-1]
        self._call_count += 1
        return LLMResponse(
            text=text,
            model="mock-model",
            provider="gemini",
            finish_reason="stop",
            prompt_tokens=50,
            completion_tokens=100,
            tokens_used=150,
        )

    async def is_available(self) -> bool:
        return True


class MockPipeline:
    """Minimal pipeline mock — returns pre-built RAGResponse per sub-query."""

    def __init__(self, responses: dict[str, RAGResponse]) -> None:
        """
        Args:
            responses: mapping of query substring → RAGResponse.
                First matching key wins. Use '' as catch-all.
        """
        self._responses = responses
        self.call_count = 0

    async def query_raw(self, request: RAGRequest) -> RAGResponse:
        self.call_count += 1
        for key, resp in self._responses.items():
            if key == "" or key in request.query:
                return resp
        raise ValueError(f"No mock response for query: {request.query!r}")


def _make_rag_response(answer: str, confidence: float = 0.8) -> RAGResponse:
    """Build a minimal valid RAGResponse for tests."""
    return RAGResponse(
        answer=answer,
        sources=[
            RetrievedChunk(
                content="Sample retrieved chunk content for testing.",
                source_file="test_doc.pdf",
                relevance_score=confidence,
            )
        ],
        timings=RAGTimings(retrieval_ms=10.0, ranking_ms=2.0, generation_ms=50.0, total_ms=62.0),
        confidence=ConfidenceScore(value=confidence, method="retrieval"),
        rag_variant="simple",
    )


def _make_planning_json(
    sub_queries: list[dict],
    reasoning: str = "Test decomposition",
    parallel_safe: bool = True,
) -> str:
    return json.dumps({
        "reasoning": reasoning,
        "parallel_safe": parallel_safe,
        "sub_queries": sub_queries,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Section 1 — ComplexityDetector
# ──────────────────────────────────────────────────────────────────────────────

def test_complexity_detector() -> None:
    section("1. ComplexityDetector — should_decompose()")

    # short queries always return False
    check("short query → False",
          not should_decompose("What is Redis?"))

    check("medium simple query → False",
          not should_decompose("How does Redis handle key expiration in production?"))

    # comparison language triggers decomposition
    check("compare keyword → True",
          should_decompose(
              "Compare the hiring costs across Engineering, Sales, and Marketing for Q3."
          ))

    check("versus keyword → True",
          should_decompose(
              "What are the attrition rates for Engineering versus Sales in the last quarter?"
          ))

    # multi-entity language
    check("both keyword → True",
          should_decompose(
              "Summarize both the Q2 and Q3 financial performance across all departments."
          ))

    # "each" alone scores +2; add "also" conjunction for +2 more → total 4, crosses threshold
    check("each keyword → True",
          should_decompose(
              "List the top performers in each department and also summarize their individual contributions as well as team impact."
          ))

    # multiple question marks
    check("multi-question → True",
          should_decompose(
              "What is the Q3 headcount? What is the Q3 attrition rate? How do these compare?"
          ))

    # long query alone adds only +1, need other signals too
    # no comparison/conjunction/multi-entity/multi-question keywords — only long (+1), below threshold
    check("long query without other signals → False",
          not should_decompose(
              "Please provide a detailed explanation of how the current onboarding process "
              "works for new employees who join the company during the fourth quarter of "
              "the fiscal year when the hiring budget has been fully committed."
          ))


# ──────────────────────────────────────────────────────────────────────────────
# Section 2 — QueryPlanner
# ──────────────────────────────────────────────────────────────────────────────

async def test_query_planner() -> None:
    section("2. QueryPlanner — decompose() + fallback")

    collections = {
        "hr_docs": "HR hiring and attrition data",
        "finance_docs": "Financial reports and budgets",
    }

    # ── 2a. Happy path — valid JSON response ──
    plan_json = _make_planning_json([
        {"query": "Q3 Engineering hiring costs", "collection": "hr_docs", "purpose": "Engineering costs"},
        {"query": "Q3 Sales hiring costs", "collection": "hr_docs", "purpose": "Sales costs"},
        {"query": "Q3 Marketing hiring costs", "collection": "hr_docs", "purpose": "Marketing costs"},
    ])
    llm = MockLLM([plan_json])
    planner = QueryPlanner(llm=llm, collections=collections)

    plan = await planner.plan(
        "Compare Q3 hiring costs across Engineering, Sales, and Marketing"
    )
    check("plan produces 3 sub-queries", len(plan.sub_queries) == 3)
    check("plan parallel_safe=True", plan.parallel_safe is True)
    check("plan reasoning non-empty", len(plan.reasoning) > 0)
    check("sub-query collections correct",
          all(sq.collection == "hr_docs" for sq in plan.sub_queries))
    check("LLM called exactly once", llm._call_count == 1)

    # ── 2b. Markdown-fenced JSON (common LLM behavior) ──
    fenced = f"```json\n{plan_json}\n```"
    llm2 = MockLLM([fenced])
    planner2 = QueryPlanner(llm=llm2, collections=collections)
    plan2 = await planner2.plan("Compare Q3 costs")
    check("fenced JSON parsed correctly", len(plan2.sub_queries) == 3)

    # ── 2c. Unparseable response → fallback plan ──
    llm3 = MockLLM(["This is completely unparseable gobbledygook!!!"])
    planner3 = QueryPlanner(llm=llm3, collections=collections)
    plan3 = await planner3.plan("Compare Q3 costs across teams")
    check("unparseable → fallback plan (1 sub-query)", len(plan3.sub_queries) == 1)
    check("fallback collection is 'default'", plan3.sub_queries[0].collection == "default")

    # ── 2d. Plan caps at 6 sub-queries ──
    big_plan_json = _make_planning_json([
        {"query": f"sub-query {i}", "collection": "hr_docs", "purpose": f"purpose {i}"}
        for i in range(10)
    ])
    llm4 = MockLLM([big_plan_json])
    planner4 = QueryPlanner(llm=llm4, collections=collections)
    plan4 = await planner4.plan("Extremely multi-part question across all teams")
    check("plan capped at 6 sub-queries", len(plan4.sub_queries) == 6)

    # ── 2e. LLM failure → AgentPlanningError ──
    class FailingLLM:
        model_name = "fail"
        provider_name = "fail"
        async def chat(self, messages, **kwargs):
            raise RuntimeError("API down")
        async def is_available(self):
            return False

    planner5 = QueryPlanner(llm=FailingLLM(), collections=collections)
    try:
        await planner5.plan("Some query")
        _fail("LLM failure → raises AgentPlanningError", "no exception raised")
    except AgentPlanningError:
        _ok("LLM failure → raises AgentPlanningError")
    except Exception as e:
        _fail("LLM failure → raises AgentPlanningError", f"wrong exception: {type(e).__name__}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 3 — ParallelRetriever
# ──────────────────────────────────────────────────────────────────────────────

async def test_parallel_retriever() -> None:
    section("3. ParallelRetriever — parallel + sequential + failure isolation")

    sub_queries = [
        SubQuery(query="Engineering Q3 hiring costs", collection="hr_docs", purpose="eng costs"),
        SubQuery(query="Sales Q3 hiring costs", collection="hr_docs", purpose="sales costs"),
        SubQuery(query="Marketing Q3 hiring costs", collection="hr_docs", purpose="mkt costs"),
    ]
    plan_parallel = DecompositionPlan(
        sub_queries=sub_queries,
        reasoning="Test parallel",
        parallel_safe=True,
    )
    plan_sequential = DecompositionPlan(
        sub_queries=sub_queries,
        reasoning="Test sequential",
        parallel_safe=False,
    )

    pipeline = MockPipeline({"": _make_rag_response("Some hiring cost answer", 0.85)})
    retriever = ParallelRetriever(pipeline=pipeline, max_concurrent=3)

    # ── 3a. Parallel execution — all succeed ──
    results = await retriever.execute(plan_parallel, parent_request_id="test-001")
    check("parallel: 3 results returned", len(results) == 3)
    check("parallel: all successful", all(r.success for r in results))
    check("parallel: pipeline called 3 times", pipeline.call_count == 3)

    # ── 3b. Sequential execution ──
    pipeline2 = MockPipeline({"": _make_rag_response("Sequential answer", 0.75)})
    retriever2 = ParallelRetriever(pipeline=pipeline2, max_concurrent=2)
    results2 = await retriever2.execute(plan_sequential, parent_request_id="test-002")
    check("sequential: 3 results returned", len(results2) == 3)
    check("sequential: all successful", all(r.success for r in results2))

    # ── 3c. Partial failure — one sub-query fails, others succeed ──
    call_n = [0]

    class PartialFailPipeline:
        async def query_raw(self, request: RAGRequest) -> RAGResponse:
            call_n[0] += 1
            if call_n[0] == 2:
                raise RuntimeError("Sub-query 2 failed")
            return _make_rag_response(f"Answer {call_n[0]}", 0.8)

    retriever3 = ParallelRetriever(pipeline=PartialFailPipeline(), max_concurrent=3)
    results3 = await retriever3.execute(plan_parallel, parent_request_id="test-003")
    check("partial failure: 3 results returned", len(results3) == 3)
    successful = [r for r in results3 if r.success]
    failed = [r for r in results3 if not r.success]
    check("partial failure: 2 succeeded", len(successful) == 2)
    check("partial failure: 1 failed", len(failed) == 1)
    check("failed result has failure_reason set", len(failed[0].failure_reason) > 0)

    # ── 3d. SubQuery.to_rag_request() carries correct IDs ──
    sq = SubQuery(query="Test query", collection="test_col", purpose="testing")
    rag_req = sq.to_rag_request("parent-req-123")
    check("to_rag_request: query preserved", rag_req.query == "Test query")
    check("to_rag_request: collection preserved", rag_req.collection_name == "test_col")
    check("to_rag_request: request_id contains parent ID",
          "parent-req-123" in rag_req.request_id)


# ──────────────────────────────────────────────────────────────────────────────
# Section 4 — ResultVerifier
# ──────────────────────────────────────────────────────────────────────────────

async def test_result_verifier() -> None:
    section("4. ResultVerifier — heuristic quality gates")

    verifier = ResultVerifier(use_llm=False)

    def _make_result(
        answer: str,
        confidence: float = 0.8,
        success: bool = True,
        failure_reason: str = "",
    ) -> SubQueryResult:
        return SubQueryResult(
            sub_query_id="test-sq-1",
            query="Test sub-query",
            collection="test_col",
            answer=answer,
            confidence=confidence,
            success=success,
            failure_reason=failure_reason,
            latency_ms=50.0,
        )

    # ── 4a. Good result passes ──
    good = _make_result(
        answer="Engineering hiring costs in Q3 were $2.1M across 45 hires, "
               "averaging $46,666 per hire including recruiter fees.",
        confidence=0.85,
    )
    results = await verifier.verify([good])
    check("good result passes verification", results[0].success)

    # ── 4b. Empty answer fails ──
    empty = _make_result(answer="   ", confidence=0.8)
    # SubQueryResult min_length check would prevent truly empty — use short answer
    short = _make_result(answer="short", confidence=0.8)
    results2 = await verifier.verify([short])
    check("short answer (<20 chars) fails verification", not results2[0].success)

    # ── 4c. Low confidence fails ──
    low_conf = _make_result(
        answer="Engineering hiring costs were approximately $2.1 million for Q3.",
        confidence=0.1,
    )
    results3 = await verifier.verify([low_conf])
    check("low confidence (<0.3) fails verification", not results3[0].success)

    # ── 4d. Non-answer phrase fails ──
    non_answer = _make_result(
        answer="I don't know the answer to this question about hiring costs.",
        confidence=0.8,
    )
    results4 = await verifier.verify([non_answer])
    check("non-answer phrase fails verification", not results4[0].success)

    # ── 4e. Pre-failed results pass through unchanged ──
    already_failed = _make_result(
        answer="",
        success=False,
        failure_reason="pipeline error",
        confidence=0.0,
    )
    # Override to bypass pydantic min_length on answer
    already_failed = SubQueryResult(
        sub_query_id="test-sq-2",
        query="Test",
        collection="col",
        answer="placeholder",
        success=False,
        failure_reason="pipeline error",
        confidence=0.0,
        latency_ms=10.0,
    )
    results5 = await verifier.verify([already_failed])
    check("pre-failed result passes through unchanged", not results5[0].success)
    check("pre-failed result failure_reason preserved",
          results5[0].failure_reason == "pipeline error")

    # ── 4f. Mixed batch ──
    mixed = [good, short, low_conf, non_answer]
    results6 = await verifier.verify(mixed)
    passed = sum(1 for r in results6 if r.success)
    check("mixed batch: only 1 passes out of 4", passed == 1)


# ──────────────────────────────────────────────────────────────────────────────
# Section 5 — AnswerSynthesizer
# ──────────────────────────────────────────────────────────────────────────────

async def test_answer_synthesizer() -> None:
    section("5. AnswerSynthesizer — synthesis call + error guards")

    synthesis_answer = (
        "In Q3, total hiring costs across departments were:\n"
        "- Engineering: $2.1M (45 hires, avg $46K)\n"
        "- Sales: $1.8M (32 hires, avg $56K)\n"
        "- Marketing: $0.9M (18 hires, avg $50K)\n"
        "Engineering had the highest absolute cost, but Sales had the highest per-hire cost."
    )

    def _make_sub_result(query: str, answer: str, success: bool = True) -> SubQueryResult:
        return SubQueryResult(
            sub_query_id=f"sq-{hash(query) % 1000}",
            query=query,
            collection="hr_docs",
            answer=answer if success else "placeholder",
            confidence=0.85 if success else 0.0,
            success=success,
            failure_reason="" if success else "pipeline error",
            latency_ms=50.0,
        )

    # ── 5a. Happy path ──
    llm = MockLLM([synthesis_answer])
    synthesizer = AnswerSynthesizer(llm=llm)
    sub_results = [
        _make_sub_result("Engineering Q3 hiring", "Engineering: $2.1M for 45 hires at avg $46K each"),
        _make_sub_result("Sales Q3 hiring", "Sales: $1.8M for 32 hires at avg $56K per hire"),
        _make_sub_result("Marketing Q3 hiring", "Marketing: $0.9M for 18 hires at avg $50K each"),
    ]
    answer = await synthesizer.synthesize(
        query="Compare Q3 hiring costs across Engineering, Sales, and Marketing",
        sub_results=sub_results,
    )
    check("synthesis returns non-empty answer", len(answer) > 0)
    check("synthesis answer matches mock response", answer == synthesis_answer)
    check("LLM called once for synthesis", llm._call_count == 1)

    # ── 5b. Failed sub-results included in prompt but synthesis still works ──
    llm2 = MockLLM(["Partial answer noting gaps where data was unavailable."])
    synthesizer2 = AnswerSynthesizer(llm=llm2)
    mixed_results = [
        _make_sub_result("Engineering Q3 hiring", "Engineering: $2.1M", success=True),
        _make_sub_result("Sales Q3 hiring", "placeholder", success=False),
    ]
    answer2 = await synthesizer2.synthesize(
        query="Compare Q3 hiring costs",
        sub_results=mixed_results,
    )
    check("synthesis with partial failures produces answer", len(answer2) > 0)

    # ── 5c. All sub-results failed → AgentSynthesisError ──
    synthesizer3 = AnswerSynthesizer(llm=MockLLM(["irrelevant"]))
    all_failed = [
        _make_sub_result("query 1", "placeholder", success=False),
        _make_sub_result("query 2", "placeholder", success=False),
    ]
    try:
        await synthesizer3.synthesize("Original query", sub_failed=all_failed)
        _fail("all-failed → raises AgentSynthesisError", "no exception raised")
    except TypeError:
        # fix: use correct kwarg
        pass
    except AgentSynthesisError:
        _ok("all-failed → raises AgentSynthesisError")

    try:
        await synthesizer3.synthesize("Original query", all_failed)
        _fail("all-failed → raises AgentSynthesisError", "no exception raised")
    except AgentSynthesisError:
        _ok("all-failed → raises AgentSynthesisError")
    except Exception as e:
        _fail("all-failed → raises AgentSynthesisError", f"wrong exception: {type(e).__name__}: {e}")

    # ── 5d. Empty synthesis output → AgentSynthesisError ──
    class EmptyLLM:
        model_name = "empty"
        provider_name = "gemini"
        async def chat(self, messages, **kwargs):
            return LLMResponse(
                text=" ",
                model="empty",
                provider="gemini",
                finish_reason="stop",
                prompt_tokens=10,
                completion_tokens=1,
                tokens_used=11,
            )

    synthesizer4 = AnswerSynthesizer(llm=EmptyLLM())
    try:
        await synthesizer4.synthesize("Query", sub_results)
        _fail("empty synthesis output → AgentSynthesisError", "no exception raised")
    except AgentSynthesisError:
        _ok("empty synthesis output → AgentSynthesisError")
    except Exception as e:
        _fail("empty synthesis output → AgentSynthesisError", f"{type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 6 — AgentOrchestrator (full flow)
# ──────────────────────────────────────────────────────────────────────────────

async def test_agent_orchestrator() -> None:
    section("6. AgentOrchestrator — full end-to-end flow with mocks")

    collections = {
        "hr_docs": "HR reports, hiring data, attrition metrics",
        "finance_docs": "Financial reports, budgets, forecasts",
    }

    plan_json = _make_planning_json([
        {"query": "Q3 Engineering hiring costs", "collection": "hr_docs",
         "purpose": "Engineering hiring total"},
        {"query": "Q3 Sales hiring costs", "collection": "hr_docs",
         "purpose": "Sales hiring total"},
        {"query": "Q3 Marketing hiring costs", "collection": "hr_docs",
         "purpose": "Marketing hiring total"},
    ])
    synthesis_answer = (
        "Q3 total hiring costs: Engineering $2.1M, Sales $1.8M, Marketing $0.9M. "
        "Engineering had the highest absolute spend."
    )
    llm = MockLLM([plan_json, synthesis_answer])
    pipeline = MockPipeline({
        "Engineering": _make_rag_response("Engineering Q3 hiring: $2.1M for 45 hires.", 0.87),
        "Sales": _make_rag_response("Sales Q3 hiring: $1.8M for 32 hires.", 0.82),
        "Marketing": _make_rag_response("Marketing Q3 hiring: $0.9M for 18 hires.", 0.79),
    })

    orchestrator = AgentOrchestrator(
        llm=llm,
        pipeline=pipeline,
        collections=collections,
        max_concurrent=3,
        use_llm_verification=False,
    )

    request = RAGRequest(
        query="Compare Q3 hiring costs across Engineering, Sales, and Marketing",
        collection_name="hr_docs",
        request_id="orch-test-001",
    )

    response = await orchestrator.execute(request)

    # ── 6a. Basic response structure ──
    check("orchestrator returns AgentResponse", isinstance(response, AgentResponse))
    check("answer is non-empty", len(response.answer) > 0)
    check("answer matches synthesis output", response.answer == synthesis_answer)
    check("plan_reasoning non-empty", len(response.plan_reasoning) > 0)

    # ── 6b. Sub-query accounting ──
    check("total_sub_queries == 3", response.total_sub_queries == 3)
    check("successful_sub_queries == 3", response.successful_sub_queries == 3)
    check("failed_sub_queries == 0", response.failed_sub_queries == 0)

    # ── 6c. Confidence computed ──
    check("confidence value > 0", response.confidence.value > 0.0)
    check("confidence method == 'agent'", response.confidence.method == "agent")

    # ── 6d. Timings populated ──
    check("total_ms > 0", response.timings.total_ms > 0)

    # ── 6e. request_id propagated ──
    check("request_id preserved", response.request_id == "orch-test-001")

    # ── 6f. All sub-results present ──
    check("sub_results has 3 entries", len(response.sub_results) == 3)
    check("all sub_results successful", all(r.success for r in response.sub_results))

    # ── 6g. AgentRetrievalError when all sub-queries fail ──
    class AlwaysFailPipeline:
        async def query_raw(self, request: RAGRequest) -> RAGResponse:
            raise RuntimeError("all pipeline calls fail")

    llm2 = MockLLM([plan_json, synthesis_answer])
    orchestrator2 = AgentOrchestrator(
        llm=llm2,
        pipeline=AlwaysFailPipeline(),
        collections=collections,
    )
    try:
        await orchestrator2.execute(request)
        _fail("all-fail pipeline → raises AgentRetrievalError", "no exception raised")
    except AgentRetrievalError:
        _ok("all-fail pipeline → raises AgentRetrievalError")
    except Exception as e:
        _fail("all-fail pipeline → raises AgentRetrievalError",
              f"wrong exception: {type(e).__name__}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 7 — Pipeline integration (configure_agents hook)
# ──────────────────────────────────────────────────────────────────────────────

async def test_pipeline_integration() -> None:
    section("7. Pipeline integration — configure_agents() + routing")

    # We test the routing logic directly without a full pipeline init
    # by checking that should_decompose() and the agent orchestrator
    # are wired correctly via configure_agents().

    # Import here to avoid top-level pipeline init
    from pipeline.rag_pipeline import RAGPipeline

    # ── 7a. configure_agents requires initialized pipeline ──
    pipeline = RAGPipeline()
    try:
        pipeline.configure_agents(collections={"docs": "test"})
        _fail("configure_agents before init → raises error", "no exception raised")
    except Exception:
        _ok("configure_agents before init → raises error")

    # ── 7b. After configure_agents, _agent_orchestrator is set ──
    # We patch _initialized to True and inject mock LLM + store
    from unittest.mock import AsyncMock, MagicMock

    mock_llm = MockLLM(["irrelevant"])
    mock_store = MagicMock()
    mock_store.initialize = AsyncMock()
    mock_store.get_collection_stats = AsyncMock(return_value={})

    pipeline2 = RAGPipeline(llm=mock_llm, store=mock_store)
    pipeline2._initialized = True  # bypass initialization

    pipeline2.configure_agents(
        collections={"hr_docs": "HR data", "finance_docs": "Finance data"},
        max_concurrent=2,
    )
    check("_agent_orchestrator set after configure_agents",
          pipeline2._agent_orchestrator is not None)
    check("_collections set correctly",
          "hr_docs" in pipeline2._collections)
    check("orchestrator is AgentOrchestrator instance",
          isinstance(pipeline2._agent_orchestrator, AgentOrchestrator))

    # ── 7c. Simple query skips agent (should_decompose=False) ──
    simple_query = RAGRequest(
        query="What is our PTO policy?",
        collection_name="hr_docs",
    )
    check("simple query should NOT decompose",
          not should_decompose(simple_query.query))

    # ── 7d. Complex query triggers agent (should_decompose=True) ──
    complex_query = RAGRequest(
        query="Compare Q3 hiring costs versus attrition costs across Engineering and Sales",
        collection_name="hr_docs",
    )
    check("complex query SHOULD decompose",
          should_decompose(complex_query.query))


# ──────────────────────────────────────────────────────────────────────────────
# Model unit tests
# ──────────────────────────────────────────────────────────────────────────────

def test_models() -> None:
    section("8. Models — SubQuery, DecompositionPlan, SubQueryResult")

    # ── SubQuery.to_rag_request() ──
    sq = SubQuery(
        query="Engineering Q3 attrition",
        collection="hr_docs",
        purpose="Get attrition data",
        variant="simple",
    )
    req = sq.to_rag_request("parent-abc")
    check("to_rag_request: query correct", req.query == "Engineering Q3 attrition")
    check("to_rag_request: collection_name correct", req.collection_name == "hr_docs")
    check("to_rag_request: variant set", req.config.rag_variant == "simple")
    check("to_rag_request: request_id has parent prefix",
          req.request_id.startswith("parent-abc::"))

    # ── SubQueryResult.from_rag_response() ──
    rag_resp = _make_rag_response("Test answer for attrition data", 0.9)
    result = SubQueryResult.from_rag_response(
        sub_query_id=sq.sub_query_id,
        query=sq.query,
        collection=sq.collection,
        response=rag_resp,
        latency_ms=123.4,
    )
    check("from_rag_response: success=True", result.success)
    check("from_rag_response: answer set", result.answer == "Test answer for attrition data")
    check("from_rag_response: confidence=0.9", abs(result.confidence - 0.9) < 0.001)
    check("from_rag_response: latency preserved", abs(result.latency_ms - 123.4) < 0.01)
    check("from_rag_response: sources populated", len(result.sources) > 0)

    # ── SubQueryResult.from_failure() ──
    failed = SubQueryResult.from_failure(
        sub_query_id="sq-fail",
        query="Some query",
        collection="col",
        reason="Pipeline timeout",
        latency_ms=500.0,
    )
    check("from_failure: success=False", not failed.success)
    check("from_failure: failure_reason set", failed.failure_reason == "Pipeline timeout")
    check("from_failure: answer is empty", failed.answer == "")

    # ── DecompositionPlan frozen (immutable) ──
    plan = DecompositionPlan(
        sub_queries=[sq],
        reasoning="Test plan",
        parallel_safe=True,
    )
    try:
        plan.reasoning = "modified"
        _fail("DecompositionPlan is frozen", "mutation succeeded")
    except Exception:
        _ok("DecompositionPlan is frozen (immutable)")


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

async def run_all() -> None:
    print("\n" + "=" * 60)
    print("  AGENT LAYER TESTS")
    print("=" * 60)

    test_complexity_detector()
    await test_query_planner()
    await test_parallel_retriever()
    await test_result_verifier()
    await test_answer_synthesizer()
    await test_agent_orchestrator()
    await test_pipeline_integration()
    test_models()

    print(f"\n{'=' * 60}")
    print(f"  Results: {_passed} passed, {_failed} failed")
    if _errors:
        print("\n  FAILURES:")
        for err in _errors:
            print(f"    • {err}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(run_all())
