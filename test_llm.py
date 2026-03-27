"""
LLM layer test — 1 API hit per provider.

1 hit covers:
    → factory creation
    → token counting   (local)
    → chat()           (1 API hit)
    → response fields
    → error handling   (local / fails before hit)
    → factory errors   (local)
"""

import asyncio
import time

from llm import (
    LLMFactory,
    BaseLLM,
    LLMResponse,
    LLMError,
    LLMAuthError,
    LLMProviderError,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
#  Config                                                             #
# ------------------------------------------------------------------ #

PROVIDERS = ["openai", "gemini"]

# 1 message — minimal tokens
MESSAGES = [
    {"role": "user", "content": "What is RAG? Reply in one sentence."}
]

CR7_MESSAGES = [
    {"role": "user", "content": "Tell me about Cristiano Ronaldo in 3 sentences."}
]

# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def separator(label: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")


def print_response(response: LLMResponse) -> None:
    print(f"  provider         : {response.provider}")
    print(f"  model            : {response.model}")
    print(f"  text             : {response.text[:150]!r}")
    print(f"  prompt_tokens    : {response.prompt_tokens}")
    print(f"  completion_tokens: {response.completion_tokens}")
    print(f"  tokens_used      : {response.tokens_used}")
    print(f"  latency_ms       : {response.latency_ms} ms")


# ------------------------------------------------------------------ #
#  Local Tests — 0 API hits                                           #
# ------------------------------------------------------------------ #

async def test_local(provider_name: str) -> BaseLLM:
    separator(f"[LOCAL] Factory + Token Count — {provider_name}")

    # Factory — 0 hits
    llm = LLMFactory.create(provider_name)
    print(f"  provider_name    : {llm.provider_name}")
    print(f"  model_name       : {llm.model_name}")

    assert isinstance(llm, BaseLLM),           "Must return BaseLLM"
    assert llm.provider_name == provider_name, "Wrong provider_name"
    assert llm.model_name,                     "model_name is empty"

    # Token counting — 0 hits (local tiktoken / native)
    prompt_text = MESSAGES[0]["content"]
    token_count = await llm.count_tokens(prompt_text)
    fits = await llm.fits_context(prompt_text, max_tokens=1000)

    print(f"  count_tokens()   : {token_count}")
    print(f"  fits_context()   : {fits}")

    assert isinstance(token_count, int), "count_tokens must return int"
    assert token_count > 0,              "token_count must be > 0"
    assert isinstance(fits, bool),       "fits_context must return bool"

    # Empty messages — 0 hits
    try:
        await llm.chat([])
        assert False, "Must raise ValueError"
    except ValueError:
        print(f"  chat([]) raised ValueError ✅")

    logger.info("[PASS] Local tests | provider=%s | tokens=%d", provider_name, token_count)
    return llm


# ------------------------------------------------------------------ #
#  1 API Hit — chat()                                                 #
# ------------------------------------------------------------------ #

async def test_one_hit(llm: BaseLLM, provider_name: str) -> None:
    separator(f"[1 HIT] chat() — {provider_name}")

    start = time.monotonic()
    response = await llm.chat(MESSAGES)                # ← ONLY API CALL
    elapsed = (time.monotonic() - start) * 1000

    print_response(response)
    print(f"\n  Total time       : {elapsed:.1f} ms")

    assert isinstance(response, LLMResponse),   "Must return LLMResponse"
    assert response.text.strip(),               "text is empty"
    assert response.provider == provider_name,  "Wrong provider"
    assert response.model,                      "model is empty"
    assert response.tokens_used >= 0,           "tokens_used must be >= 0"
    assert response.latency_ms >= 0,            "latency_ms must be >= 0"
    assert response.prompt_tokens >= 0,         "prompt_tokens must be >= 0"
    assert response.completion_tokens >= 0,     "completion_tokens must be >= 0"

    logger.info(
        "[PASS] 1 hit chat | provider=%s | tokens=%d | latency=%.1f ms",
        provider_name,
        response.tokens_used,
        response.latency_ms,
    )


# ------------------------------------------------------------------ #
#  Error Tests — 0 API hits (fails before hitting API)               #
# ------------------------------------------------------------------ #

def test_errors(provider_name: str) -> None:
    separator(f"[ERRORS] — {provider_name}")

    # Invalid key — fails before API hit
    try:
        llm = LLMFactory.create(provider_name, api_key="invalid-key-123")
        llm.chat(MESSAGES)
        assert False, "Must raise LLMAuthError"
    except (LLMAuthError, LLMError) as e:
        print(f"  ✅ Bad key raised {type(e).__name__}")

    # Unknown provider — local only
    try:
        LLMFactory.create("unknown")
        assert False, "Must raise LLMProviderError"
    except LLMProviderError as e:
        print(f"  ✅ Unknown provider raised LLMProviderError")

    # Empty provider — local only
    try:
        LLMFactory.create("")
        assert False, "Must raise LLMProviderError"
    except LLMProviderError as e:
        print(f"  ✅ Empty provider raised LLMProviderError")

    logger.info("[PASS] Error tests | provider=%s", provider_name)

async def test_real_call(llm: BaseLLM, provider_name: str) -> None:
    separator(f"[REAL CALL] Cristiano Ronaldo — {provider_name}")

    print(f"  Prompt : {CR7_MESSAGES[0]['content']!r}")
    print(f"  Calling {provider_name}...\n")

    start = time.monotonic()
    response = await llm.chat(CR7_MESSAGES)
    elapsed = (time.monotonic() - start) * 1000

    print(f"  ─── Response ───────────────────────────────────────")
    print(f"  {response.text}")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  model            : {response.model}")
    print(f"  prompt_tokens    : {response.prompt_tokens}")
    print(f"  completion_tokens: {response.completion_tokens}")
    print(f"  tokens_used      : {response.tokens_used}")
    print(f"  latency_ms       : {elapsed:.1f} ms")

    assert response.text.strip(), "Response is empty"
    logger.info(
        "[PASS] Real call | provider=%s | tokens=%d | latency=%.1f ms",
        provider_name,
        response.tokens_used,
        elapsed,
    )


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

async def main() -> None:
    separator("LLM LAYER TESTS — 1 API HIT PER PROVIDER")
    print(f"  Providers : {PROVIDERS}")
    print(f"  API hits  : {len(PROVIDERS)} total")

    for provider_name in PROVIDERS:
        separator(f"PROVIDER: {provider_name.upper()}")
        try:
            # 0 hits — local only
            llm = await test_local(provider_name)

            # 1 hit — only API call
            await test_one_hit(llm, provider_name)

            # ✅ Real call — Cristiano Ronaldo
            await test_real_call(llm, provider_name)
            
            # 0 hits — fails before API
            await test_errors(provider_name)

            separator(f"✅ ALL PASSED — {provider_name.upper()}")
            logger.info("All tests passed | provider=%s", provider_name)

        except AssertionError as e:
            logger.error("[FAIL] | provider=%s | error=%s", provider_name, e)
            print(f"\n  ❌ FAILED: {e}")

        except Exception as e:
            logger.error("[FAIL] | provider=%s | error=%s", provider_name, e)
            print(f"\n  ❌ ERROR: {type(e).__name__} | {e}")

        print("\n")


if __name__ == "__main__":
    asyncio.run(main())