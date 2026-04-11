"""
End-to-end LLM provider pipeline tests — real API calls plus local validation.

Test scope:
    Phases 1-4 are pure local tests (0 API calls) covering LLMResponse model
    validation, exception hierarchy, constants/exports, and factory error paths.
    Phases 5-9 make 3 real API calls per provider (generate, chat, is_available)
    and validate responses, token counts, finish reasons, and bad-key handling.

Flow:
    Local tests (Phases 1-4) → per-provider loop (Phases 5-9: local setup,
    generate, chat, is_available, bad key) → summary.

Dependencies:
    GEMINI_API_KEY and OPENAI_API_KEY set in .env, network access to both APIs.
    Total cost: ~70 tokens per provider (~140 tokens total).
"""

import asyncio
import sys
import time

from pydantic import ValidationError

from llm import (
    LLMFactory,
    BaseLLM,
    LLMResponse,
    LLMError,
    LLMAuthError,
    LLMProviderError,
    SUPPORTED_PROVIDERS,
    VALID_FINISH_REASONS,
)
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# Config — minimal prompts to save tokens

PROVIDERS = ["gemini", "openai"]

# Single-turn prompt — short input, capped output
GENERATE_PROMPT = "Say hi"
GENERATE_MAX_TOKENS = 50

# Multi-turn prompt — tests message conversion
CHAT_MESSAGES = [
    {"role": "user", "content": "Say hello in one word."},
]
CHAT_MAX_TOKENS = 50

# Counters and helpers

_pass_count = 0
_fail_count = 0
_skip_count = 0
_api_calls = 0


def separator(label: str) -> None:
    """Print a visual section separator."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


def _report(test_name: str, passed: bool, detail: str = "") -> None:
    """Log individual test result and update counters."""
    global _pass_count, _fail_count
    if passed:
        _pass_count += 1
        print(f"  [PASS] {test_name} {detail}")
    else:
        _fail_count += 1
        print(f"  [FAIL] {test_name} {detail}")
        logger.error("[FAIL] %s %s", test_name, detail)


def _skip(test_name: str, reason: str) -> None:
    """Log a skipped test."""
    global _skip_count
    _skip_count += 1
    print(f"  [SKIP] {test_name} | {reason}")


def _track_api_call() -> None:
    """Increment API call counter for budget tracking."""
    global _api_calls
    _api_calls += 1


def print_response(response: LLMResponse) -> None:
    """Pretty-print an LLMResponse for visual inspection."""
    print(f"    provider         : {response.provider}")
    print(f"    model            : {response.model}")
    print(f"    finish_reason    : {response.finish_reason}")
    print(f"    text             : {response.text[:120]!r}")
    print(f"    prompt_tokens    : {response.prompt_tokens}")
    print(f"    completion_tokens: {response.completion_tokens}")
    print(f"    tokens_used      : {response.tokens_used}")
    print(f"    latency_ms       : {response.latency_ms:.1f} ms")
    print(f"    cached           : {response.cached}")


# Phase 1: LLMResponse model validation — 0 API calls

def test_response_model() -> None:
    """Verifies LLMResponse construction, immutability, and field validators."""
    separator("Phase 1: LLMResponse Model Validation (0 API calls)")

    # Valid construction
    resp = LLMResponse(
        text="Hello",
        model="test-model",
        provider="openai",
        finish_reason="stop",
        prompt_tokens=5,
        completion_tokens=3,
        tokens_used=8,
        latency_ms=100.0,
    )
    _report(
        "valid_construction",
        resp.text == "Hello"
        and resp.model == "test-model"
        and resp.provider == "openai"
        and resp.finish_reason == "stop"
        and resp.tokens_used == 8,
    )

    # Frozen immutability
    try:
        resp.text = "modified"
        _report("frozen_immutability", False, "| mutation allowed")
    except (ValidationError, AttributeError, TypeError):
        _report("frozen_immutability", True)

    # Empty text rejected
    try:
        LLMResponse(text="", model="m", provider="openai")
        _report("empty_text_rejected", False, "| accepted empty text")
    except (ValidationError, ValueError):
        _report("empty_text_rejected", True)

    # Whitespace-only text rejected
    try:
        LLMResponse(text="   \n\t", model="m", provider="openai")
        _report("whitespace_text_rejected", False, "| accepted whitespace")
    except (ValidationError, ValueError):
        _report("whitespace_text_rejected", True)

    # Unsupported provider rejected
    try:
        LLMResponse(text="hi", model="m", provider="anthropic")
        _report("unsupported_provider_rejected", False, "| accepted 'anthropic'")
    except (ValidationError, ValueError):
        _report("unsupported_provider_rejected", True)

    # Provider case normalization
    resp2 = LLMResponse(
        text="hi", model="m", provider="OpenAI",
        prompt_tokens=1, completion_tokens=1, tokens_used=2,
    )
    _report("provider_case_normalized", resp2.provider == "openai")

    # Negative tokens rejected
    try:
        LLMResponse(
            text="hi", model="m", provider="openai", prompt_tokens=-1,
        )
        _report("negative_tokens_rejected", False)
    except (ValidationError, ValueError):
        _report("negative_tokens_rejected", True)

    # Token consistency — tokens_used < prompt + completion
    try:
        LLMResponse(
            text="hi", model="m", provider="openai",
            prompt_tokens=10, completion_tokens=5, tokens_used=12,
        )
        _report("token_consistency_enforced", False, "| accepted inconsistent")
    except (ValidationError, ValueError):
        _report("token_consistency_enforced", True)

    # Token overhead allowed — tokens_used > prompt + completion
    resp3 = LLMResponse(
        text="hi", model="m", provider="openai",
        prompt_tokens=10, completion_tokens=5, tokens_used=20,
    )
    _report("token_overhead_allowed", resp3.tokens_used == 20)

    # finish_reason defaults to "unknown"
    resp4 = LLMResponse(
        text="hi", model="m", provider="openai",
        prompt_tokens=1, completion_tokens=1, tokens_used=2,
    )
    _report("finish_reason_default_unknown", resp4.finish_reason == "unknown")

    # finish_reason case normalization
    resp5 = LLMResponse(
        text="hi", model="m", provider="openai",
        finish_reason="STOP",
        prompt_tokens=1, completion_tokens=1, tokens_used=2,
    )
    _report("finish_reason_normalized", resp5.finish_reason == "stop")

    # cached defaults to False
    _report("cached_default_false", resp.cached is False)

    # metadata defaults to empty dict
    _report("metadata_default_empty", resp.metadata == {})

    # metadata preserved
    meta = {"safety": "high", "logprobs": [0.1]}
    resp6 = LLMResponse(
        text="hi", model="m", provider="openai",
        metadata=meta,
        prompt_tokens=1, completion_tokens=1, tokens_used=2,
    )
    _report("metadata_preserved", resp6.metadata == meta)

    # Negative latency rejected
    try:
        LLMResponse(
            text="hi", model="m", provider="openai", latency_ms=-1.0,
        )
        _report("negative_latency_rejected", False)
    except (ValidationError, ValueError):
        _report("negative_latency_rejected", True)


# Phase 2: Exception hierarchy — 0 API calls

def test_exception_hierarchy() -> None:
    """Verifies that all LLM exception subclasses inherit from LLMError."""
    separator("Phase 2: Exception Hierarchy (0 API calls)")

    # All subclasses inherit from LLMError
    subclasses = [
        LLMAuthError, LLMProviderError,
    ]
    # Import the rest directly for thorough check
    from llm.exceptions.llm_exceptions import (
        LLMRateLimitError,
        LLMTimeoutError,
        LLMTokenLimitError,
    )
    subclasses.extend([LLMRateLimitError, LLMTimeoutError, LLMTokenLimitError])

    all_inherit = all(issubclass(cls, LLMError) for cls in subclasses)
    _report("all_inherit_from_llm_error", all_inherit)

    # LLMError inherits from Exception
    _report("llm_error_is_exception", issubclass(LLMError, Exception))

    # Catch-all works
    all_caught = True
    for cls in subclasses:
        try:
            raise cls("test")
        except LLMError:
            pass
        except Exception:
            all_caught = False
            break
    _report("catch_all_works", all_caught)

    # Message preserved
    exc = LLMAuthError("bad key")
    _report("message_preserved", str(exc) == "bad key")


# Phase 3: Constants and exports — 0 API calls

def test_constants() -> None:
    """Verifies module-level constants and __init__ exports are complete."""
    separator("Phase 3: Constants & Exports (0 API calls)")

    _report(
        "supported_providers_complete",
        "openai" in SUPPORTED_PROVIDERS and "gemini" in SUPPORTED_PROVIDERS,
    )
    _report(
        "valid_finish_reasons_complete",
        {"stop", "length", "safety", "unknown"}.issubset(VALID_FINISH_REASONS),
    )
    _report(
        "factory_available_matches",
        set(LLMFactory.available_providers()) == set(SUPPORTED_PROVIDERS),
    )


# Phase 4: Factory — 0 API calls (error paths)

def test_factory_errors() -> None:
    """Verifies factory error handling for unknown, empty, and invalid inputs."""
    separator("Phase 4: Factory Error Handling (0 API calls)")

    # Unknown provider
    try:
        LLMFactory.create("anthropic")
        _report("unknown_provider_rejected", False)
    except LLMProviderError:
        _report("unknown_provider_rejected", True)

    # Empty provider
    try:
        LLMFactory.create("")
        _report("empty_provider_rejected", False)
    except LLMProviderError:
        _report("empty_provider_rejected", True)

    # Whitespace-only provider
    try:
        LLMFactory.create("   ")
        _report("whitespace_provider_rejected", False)
    except LLMProviderError:
        _report("whitespace_provider_rejected", True)

    # Register invalid class
    try:
        LLMFactory.register("bad", str)
        _report("invalid_class_rejected", False)
    except LLMProviderError:
        _report("invalid_class_rejected", True)

    # Register + create round-trip with a simple mock
    class _TinyMock(BaseLLM):
        """Minimal BaseLLM for factory registration test."""

        @property
        def provider_name(self) -> str:
            return "tiny"

        @property
        def model_name(self) -> str:
            return "tiny-v1"

        async def generate(self, prompt, **kw):
            return LLMResponse(
                text="ok", model="tiny-v1", provider="gemini",
                finish_reason="stop",
                prompt_tokens=1, completion_tokens=1, tokens_used=2,
            )

        async def chat(self, messages, **kw):
            return await self.generate("")

        async def count_tokens(self, text):
            return len(text.split())

        async def is_available(self):
            return True

    try:
        LLMFactory.register("_tiny_test", _TinyMock)
        provider = LLMFactory.create("_tiny_test")
        passed = (
            isinstance(provider, BaseLLM)
            and provider.provider_name == "tiny"
        )
        _report("register_and_create_roundtrip", passed)
    finally:
        LLMFactory._registry.pop("_tiny_test", None)


# Phase 5: Per-provider local tests — 0 API calls

async def test_provider_local(provider_name: str) -> BaseLLM:
    """Verifies factory creation, properties, and token counting without generation calls.

    Note: Gemini count_tokens makes an API call internally, but it is a
    lightweight metadata call with no output tokens.

    Args:
        provider_name: Provider to test.

    Returns:
        Created BaseLLM instance for reuse in API tests.
    """
    separator(f"Phase 5: Local Tests — {provider_name.upper()} (0 generation calls)")

    # Factory creation
    llm = LLMFactory.create(provider_name)
    _report(
        "factory_creates_base_llm",
        isinstance(llm, BaseLLM),
    )

    # Properties
    _report("provider_name_correct", llm.provider_name == provider_name)
    _report("model_name_not_empty", bool(llm.model_name))
    print(f"    provider_name  : {llm.provider_name}")
    print(f"    model_name     : {llm.model_name}")

    # repr includes class info
    repr_str = repr(llm)
    _report(
        "repr_contains_info",
        provider_name in repr_str and llm.model_name in repr_str,
    )
    print(f"    repr           : {repr_str}")

    # Token counting — local for OpenAI (tiktoken), API metadata call for Gemini
    test_text = "What is RAG"
    token_count = await llm.count_tokens(test_text)
    _report("count_tokens_returns_int", isinstance(token_count, int))
    _report("count_tokens_positive", token_count > 0)
    print(f"    count_tokens   : {token_count} (for {test_text!r})")

    # Empty text returns 0
    empty_count = await llm.count_tokens("")
    _report("count_tokens_empty_is_zero", empty_count == 0)

    # fits_context — reuses count_tokens internally, no extra API call
    fits = await llm.fits_context(test_text, max_tokens=1000)
    _report("fits_context_true_for_short", fits is True)

    does_not_fit = await llm.fits_context(test_text, max_tokens=1)
    _report("fits_context_false_for_tiny_budget", does_not_fit is False)

    # Empty messages raise ValueError — no API call
    try:
        await llm.chat([])
        _report("empty_chat_raises_value_error", False)
    except ValueError:
        _report("empty_chat_raises_value_error", True)

    return llm


# Phase 6: generate() — 1 API call per provider

async def test_generate(llm: BaseLLM, provider_name: str) -> None:
    """Verifies generate() response structure, token counts, and finish reason.

    Args:
        llm: Provider instance.
        provider_name: Provider name for assertions.
    """
    separator(f"Phase 6: generate() — {provider_name.upper()} (1 API call)")

    _track_api_call()
    start = time.monotonic()
    response = await llm.generate(
        GENERATE_PROMPT,
        max_tokens=GENERATE_MAX_TOKENS,
    )
    elapsed = (time.monotonic() - start) * 1000

    print_response(response)
    print(f"    wall_clock_ms    : {elapsed:.1f}")

    # Type check
    _report("returns_llm_response", isinstance(response, LLMResponse))

    # Text not empty
    _report("text_not_empty", bool(response.text.strip()))

    # Provider matches
    _report("provider_matches", response.provider == provider_name)

    # Model not empty
    _report("model_not_empty", bool(response.model))

    # finish_reason is a known value
    _report(
        "finish_reason_is_known",
        response.finish_reason in VALID_FINISH_REASONS,
        f"| got {response.finish_reason!r}",
    )

    # finish_reason is lowercase
    _report(
        "finish_reason_is_lowercase",
        response.finish_reason == response.finish_reason.lower(),
    )

    # Token counts are non-negative
    _report("prompt_tokens_non_negative", response.prompt_tokens >= 0)
    _report("completion_tokens_non_negative", response.completion_tokens >= 0)
    _report("tokens_used_non_negative", response.tokens_used >= 0)

    # Token consistency — tokens_used >= prompt + completion
    expected_min = response.prompt_tokens + response.completion_tokens
    _report(
        "token_consistency",
        response.tokens_used >= expected_min,
        f"| used={response.tokens_used} >= prompt({response.prompt_tokens})+comp({response.completion_tokens})",
    )

    # Latency is positive
    _report("latency_positive", response.latency_ms > 0)

    # cached is False (fresh call)
    _report("cached_is_false", response.cached is False)

    logger.info(
        "[PASS] generate | provider=%s | tokens=%d | latency=%.1f ms",
        provider_name,
        response.tokens_used,
        response.latency_ms,
    )


# Phase 7: chat() — 1 API call per provider

async def test_chat(llm: BaseLLM, provider_name: str) -> None:
    """Verifies chat() response structure including prompt_tokens > 0.

    Args:
        llm: Provider instance.
        provider_name: Provider name for assertions.
    """
    separator(f"Phase 7: chat() — {provider_name.upper()} (1 API call)")

    _track_api_call()
    start = time.monotonic()
    response = await llm.chat(
        CHAT_MESSAGES,
        max_tokens=CHAT_MAX_TOKENS,
    )
    elapsed = (time.monotonic() - start) * 1000

    print_response(response)
    print(f"    wall_clock_ms    : {elapsed:.1f}")

    # Core assertions — same as generate
    _report("returns_llm_response", isinstance(response, LLMResponse))
    _report("text_not_empty", bool(response.text.strip()))
    _report("provider_matches", response.provider == provider_name)
    _report("model_not_empty", bool(response.model))
    _report(
        "finish_reason_valid",
        response.finish_reason in VALID_FINISH_REASONS,
        f"| got {response.finish_reason!r}",
    )
    _report("tokens_used_non_negative", response.tokens_used >= 0)
    _report("latency_positive", response.latency_ms > 0)

    # Chat-specific: verify prompt tokens > 0 (message was sent)
    _report(
        "prompt_tokens_positive",
        response.prompt_tokens > 0,
        f"| got {response.prompt_tokens}",
    )

    logger.info(
        "[PASS] chat | provider=%s | tokens=%d | latency=%.1f ms",
        provider_name,
        response.tokens_used,
        response.latency_ms,
    )


# Phase 8: is_available() — 1 API call per provider

async def test_is_available(llm: BaseLLM, provider_name: str) -> None:
    """Verifies that is_available() returns True for a working provider.

    Args:
        llm: Provider instance.
        provider_name: Provider name for logging.
    """
    separator(f"Phase 8: is_available() — {provider_name.upper()} (1 API call)")

    _track_api_call()
    available = await llm.is_available()

    _report("is_available_returns_bool", isinstance(available, bool))
    _report("provider_is_available", available is True)
    print(f"    is_available   : {available}")

    logger.info("[PASS] is_available | provider=%s | available=%s", provider_name, available)


# Phase 9: Error handling with bad API key — 0 API calls

async def test_bad_api_key(provider_name: str) -> None:
    """Verifies that a bad API key raises LLMError (never a raw SDK error).

    Some providers validate the key format locally (0 API calls).
    Others send the request and get a 401 (1 API call with invalid key).
    Either way, the error must be wrapped in an LLMError subclass.

    Args:
        provider_name: Provider to test.
    """
    separator(f"Phase 9: Bad API Key — {provider_name.upper()}")

    try:
        bad_llm = LLMFactory.create(provider_name, api_key="invalid-key-12345")
        # Some providers accept construction but fail on first call
        await bad_llm.chat(
            [{"role": "user", "content": "test"}],
            max_tokens=1,
        )
        _report("bad_key_raises_error", False, "| no error raised")
    except LLMAuthError:
        _report("bad_key_raises_auth_error", True)
    except LLMError as exc:
        # Acceptable — some providers wrap auth failures differently
        _report(
            "bad_key_raises_llm_error",
            True,
            f"| got {type(exc).__name__}",
        )
    except Exception as exc:
        # NOT acceptable — raw SDK error leaked through
        _report(
            "bad_key_raises_error",
            False,
            f"| leaked raw {type(exc).__name__}: {exc}",
        )


# Main runner

async def main() -> None:
    """Run all test phases sequentially.

    Phases 1-4 are pure local (0 API calls).
    Phases 5-9 run per provider (3 API calls each).
    """
    global _pass_count, _fail_count, _skip_count, _api_calls

    overall_start = time.monotonic()

    separator("LLM LAYER TEST SUITE — REAL API CALLS")
    print(f"  Providers        : {PROVIDERS}")
    print(f"  Budget per provider: 3 API calls (~150 tokens)")
    print(f"  Total budget     : {len(PROVIDERS) * 3} API calls (~{len(PROVIDERS) * 150} tokens)")

    # Pure local tests (0 API calls)

    test_response_model()
    test_exception_hierarchy()
    test_constants()
    test_factory_errors()

    # Per-provider tests (3 API calls each)

    for provider_name in PROVIDERS:
        separator(f"PROVIDER: {provider_name.upper()}")

        # Check if API key is available
        key_attr = f"{provider_name}_api_key"
        api_key = getattr(settings, key_attr, None)

        if not api_key:
            _skip(
                f"all_{provider_name}_tests",
                f"No API key — set {key_attr.upper()} in .env",
            )
            continue

        try:
            # Phase 5: Local tests (0 generation API calls)
            llm = await test_provider_local(provider_name)

            # Phase 6: generate() — 1 API call
            await test_generate(llm, provider_name)

            # Phase 7: chat() — 1 API call
            await test_chat(llm, provider_name)

            # Phase 8: is_available() — 1 API call
            await test_is_available(llm, provider_name)

            # Phase 9: Bad API key — 0 or 1 API call
            await test_bad_api_key(provider_name)

            separator(f"PROVIDER {provider_name.upper()} — COMPLETE")

        except AssertionError as exc:
            logger.error("[FAIL] %s | assertion=%s", provider_name, exc)
            print(f"\n  ASSERTION FAILED: {exc}")

        except Exception as exc:
            logger.error(
                "[FAIL] %s | unexpected=%s | error=%s",
                provider_name,
                type(exc).__name__,
                exc,
            )
            print(f"\n  UNEXPECTED ERROR: {type(exc).__name__} | {exc}")

    # Summary

    elapsed = (time.monotonic() - overall_start) * 1000

    separator("TEST SUMMARY")
    print(f"  Passed           : {_pass_count}")
    print(f"  Failed           : {_fail_count}")
    print(f"  Skipped          : {_skip_count}")
    print(f"  API calls made   : {_api_calls}")
    print(f"  Total time       : {elapsed:.0f} ms")
    print()

    if _fail_count > 0:
        logger.error(
            "TEST SUITE FAILED | passed=%d | failed=%d | skipped=%d | api_calls=%d | elapsed=%.0f ms",
            _pass_count,
            _fail_count,
            _skip_count,
            _api_calls,
            elapsed,
        )
        sys.exit(1)
    else:
        logger.info(
            "TEST SUITE PASSED | passed=%d | failed=%d | skipped=%d | api_calls=%d | elapsed=%.0f ms",
            _pass_count,
            _fail_count,
            _skip_count,
            _api_calls,
            elapsed,
        )
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
