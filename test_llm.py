"""
Real LLM test — live API calls to all three Groq models.

Tests each model with:
    1. generate()     — single-turn prompt
    2. chat()         — multi-turn messages
    3. count_tokens() — local token count (no API hit)

Models under test:
    GROQ_MODEL_FAST     llama-3.1-8b-instant      classify / decompose
    GROQ_MODEL_STRONG   llama-3.3-70b-versatile   final synthesis
    GROQ_MODEL_FALLBACK qwen/qwen3-32b             fallback if strong hits 429

Run:
    python test_llm.py
"""

import asyncio
import time

from llm import LLMFactory, LLMResponse
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Models to test ────────────────────────────────────────────────────────────
GROQ_MODELS = [
    (settings.GROQ_MODEL_FAST,     "Fast     — classify / decompose / simple queries (RPD: 14,400)"),
    (settings.GROQ_MODEL_STRONG,   "Strong   — final answer synthesis, complex queries (RPD: 1,000)"),
    (settings.GROQ_MODEL_FALLBACK, "Fallback — if strong hits 429                      (RPD: 1,000, RPM: 60)"),
]

GENERATE_PROMPT = "What is Retrieval-Augmented Generation? Answer in one sentence."

CHAT_MESSAGES = [
    {"role": "system", "content": "You are a concise AI assistant."},
    {"role": "user",   "content": "Name the three laws of robotics. One line each."},
]

TOKEN_COUNT_TEXT = "The quick brown fox jumps over the lazy dog."


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def _section(title: str) -> None:
    print(f"\n{'-' * 70}")
    print(f"  {title}")
    print(f"{'-' * 70}")


def _print_response(label: str, response: LLMResponse, elapsed_ms: float) -> None:
    print(f"  [{label}]")
    print(f"  model             : {response.model}")
    print(f"  answer            : {response.text.strip()[:200]}")
    print(f"  prompt_tokens     : {response.prompt_tokens}")
    print(f"  completion_tokens : {response.completion_tokens}")
    print(f"  tokens_used       : {response.tokens_used}")
    print(f"  latency_ms        : {elapsed_ms:.1f} ms")


# ── Per-model test ────────────────────────────────────────────────────────────

async def test_groq_model(model: str, role_desc: str) -> bool:
    """Run generate, chat, and count_tokens against one Groq model.

    Returns:
        True if all calls succeeded, False if any failed.
    """
    _banner(f"GROQ | {model}")
    print(f"  Role : {role_desc}")

    try:
        llm = LLMFactory.create("groq", model=model)
        print(f"  Provider : {llm.provider_name} | Model : {llm.model_name}\n")

        # ── 1. generate() ─────────────────────────────────────────────────────
        _section("1. generate()")
        print(f"  Prompt : {GENERATE_PROMPT!r}\n")

        t0 = time.perf_counter()
        gen_response = await llm.generate(GENERATE_PROMPT)
        gen_ms = (time.perf_counter() - t0) * 1000

        _print_response("generate", gen_response, gen_ms)

        # ── 2. chat() ─────────────────────────────────────────────────────────
        _section("2. chat()")
        for msg in CHAT_MESSAGES:
            print(f"  [{msg['role']}] {msg['content']}")
        print()

        t0 = time.perf_counter()
        chat_response = await llm.chat(CHAT_MESSAGES)
        chat_ms = (time.perf_counter() - t0) * 1000

        _print_response("chat", chat_response, chat_ms)

        # ── 3. count_tokens() — no API hit ────────────────────────────────────
        _section("3. count_tokens()  [local, no API hit]")
        print(f"  Text : {TOKEN_COUNT_TEXT!r}")

        token_count = await llm.count_tokens(TOKEN_COUNT_TEXT)
        print(f"  Token count : {token_count}")

        _banner(f"PASS — {model}")
        logger.info(
            "Groq model passed | model=%s | gen_ms=%.1f | chat_ms=%.1f | tokens=%d",
            model, gen_ms, chat_ms, token_count,
        )
        return True

    except Exception as exc:
        _banner(f"FAIL — {model}")
        print(f"  Error : {type(exc).__name__} | {exc}")
        logger.error("Groq model failed | model=%s | error=%s", model, exc)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

async def run() -> None:
    _banner("GROQ LLM TEST — Real API Calls (All Three Models)")
    print(f"""
  Models under test:
    FAST     : {settings.GROQ_MODEL_FAST}
    STRONG   : {settings.GROQ_MODEL_STRONG}
    FALLBACK : {settings.GROQ_MODEL_FALLBACK}

  Calls per model : 2 API hits (generate + chat) + 1 local (count_tokens)
  Total API hits  : {len(GROQ_MODELS) * 2}
    """)

    results: list[tuple[str, bool]] = []

    for model, role_desc in GROQ_MODELS:
        passed = await test_groq_model(model, role_desc)
        results.append((model, passed))

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("SUMMARY")
    all_passed = True
    for model, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {model}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  All Groq models are working.")
    else:
        print("  Some models failed — check errors above.")

    logger.info("Groq test complete | all_passed=%s", all_passed)


if __name__ == "__main__":
    asyncio.run(run())
