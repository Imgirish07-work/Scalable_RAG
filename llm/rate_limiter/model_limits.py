"""
Per-model rate limit registry for the LLM rate limiter.

Design:
    Module-level registry dict (MODEL_RATE_LIMITS) maps exact model ID strings
    to frozen _ModelLimits dataclasses. get_rate_limit_config() looks up the
    model and returns a RateLimiterConfig ready for LLMRateLimiter.

    Only models that are actively used appear here — Groq pool models and
    the Gemini fallback. All limits are sourced directly from official provider
    documentation (Groq free-tier rate limits page, Gemini AI Studio limits)
    and are NOT tunable via settings — setting a higher value than the provider
    enforces causes 429s.

    pool: Marks which Groq model pool(s) a model belongs to:
        "FAST"   — lightweight tasks (eval, rewrite, classify) max_tokens≤512
        "STRONG" — final synthesis / complex generation  max_tokens>512 or None
        "BOTH"   — shared budget; model appears in both pools (e.g. qwen3-32b)
        None     — non-Groq provider; pool routing does not apply

    Adding a new Groq model requires exactly one line in MODEL_RATE_LIMITS
    plus an entry in _ALL_POOL_MODELS (groq_model_pool.py) and the appropriate
    pool list in model_router.py.

Chain of Responsibility:
    LLMFactory.create_rate_limited() calls get_rate_limit_config(model_name) →
    returns RateLimiterConfig → passed to LLMRateLimiter constructor.

    GroqModelPool reads MODEL_RATE_LIMITS directly via get_model_limits() to
    build its per-model RateLimitTracker budget.

Sources:
    Groq: https://console.groq.com/docs/rate-limits  (free tier, April 2026)
    Gemini: https://ai.google.dev/gemini-api/docs/rate-limits

Dependencies:
    llm.rate_limiter.rate_limiter_config.RateLimiterConfig, utils.logger.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from llm.rate_limiter.rate_limiter_config import RateLimiterConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Conservative fallback limits used for models not found in the registry.
# Deliberately low so unknown models degrade gracefully rather than causing 429 floods.
_UNKNOWN_MODEL_RPM = 10
_UNKNOWN_MODEL_RPD = 200
_UNKNOWN_MODEL_TPM = 10_000
_UNKNOWN_MODEL_TPD = 100_000

# Pool membership literals — used by ModelRouter to select candidates
PoolTag = Literal["FAST", "STRONG", "BOTH"]


@dataclass(frozen=True)
class _ModelLimits:
    """Provider-enforced hard limits for one model across all 4 rate limit dimensions.

    All four dimensions can independently be the binding constraint for any
    given request. The ModelRouter checks all four before dispatching.

    Attributes:
        rpm: Maximum requests per minute (resets every minute server-side).
        rpd: Maximum requests per day (resets at midnight UTC server-side).
        tpm: Maximum tokens per minute (prompt + completion combined).
        tpd: Maximum tokens per day (prompt + completion combined).
        pool: Groq pool membership tag, or None for non-Groq providers.
            "FAST"   — lightweight tasks (eval, rewrite); max_tokens≤512
            "STRONG" — final synthesis / complex reasoning
            "BOTH"   — model shared between both pools; one budget tracked
    """

    rpm: int                        # requests per minute
    rpd: int                        # requests per day
    tpm: int                        # tokens per minute
    tpd: int                        # tokens per day
    pool: Optional[PoolTag] = None  # Groq pool tag; None = not a Groq pool model


# Registry — only active models are listed here

MODEL_RATE_LIMITS: dict[str, _ModelLimits] = {

    # Groq — FAST pool
    "llama-3.1-8b-instant":                      _ModelLimits(rpm=30,  rpd=14_400, tpm=6_000,  tpd=500_000, pool="FAST"),

    # Groq — STRONG pool
    "moonshotai/kimi-k2-instruct":               _ModelLimits(rpm=60,  rpd=1_000,  tpm=10_000, tpd=300_000, pool="STRONG"),
    "llama-3.3-70b-versatile":                   _ModelLimits(rpm=30,  rpd=1_000,  tpm=12_000, tpd=100_000, pool="STRONG"),
    "meta-llama/llama-4-scout-17b-16e-instruct": _ModelLimits(rpm=30,  rpd=1_000,  tpm=30_000, tpd=500_000, pool="STRONG"),

    # Groq — BOTH pools (shared budget)
    "qwen/qwen3-32b":                            _ModelLimits(rpm=60,  rpd=1_000,  tpm=6_000,  tpd=500_000, pool="BOTH"),

    # Gemini — fallback provider
    "gemini-2.5-flash":                          _ModelLimits(rpm=10,  rpd=500,    tpm=250_000, tpd=1_000_000),
}


def get_model_limits(model_name: str) -> _ModelLimits:
    """Return the _ModelLimits for the given model, with a safe fallback.

    Used by RateLimitTracker to initialize per-model budget ceilings and
    by ModelRouter to check 4-dimension headroom before dispatching.

    Args:
        model_name: Exact model ID string e.g. 'llama-3.1-8b-instant'.

    Returns:
        _ModelLimits for the model, or conservative fallback limits if not found.
    """
    limits = MODEL_RATE_LIMITS.get(model_name)

    if limits is None:
        logger.warning(
            "No rate limits registered for model '%s'. "
            "Using conservative fallback (rpm=%d, rpd=%d, tpm=%d, tpd=%d). "
            "Add this model to MODEL_RATE_LIMITS in "
            "llm/rate_limiter/model_limits.py to silence this warning.",
            model_name,
            _UNKNOWN_MODEL_RPM,
            _UNKNOWN_MODEL_RPD,
            _UNKNOWN_MODEL_TPM,
            _UNKNOWN_MODEL_TPD,
        )
        return _ModelLimits(
            rpm=_UNKNOWN_MODEL_RPM,
            rpd=_UNKNOWN_MODEL_RPD,
            tpm=_UNKNOWN_MODEL_TPM,
            tpd=_UNKNOWN_MODEL_TPD,
        )

    return limits


def get_rate_limit_config(
    model_name: str,
    max_concurrent: int = 5,
    burst_multiplier: float = 1.0,
) -> RateLimiterConfig:
    """Return a RateLimiterConfig with the correct RPM/RPD for the given model.

    Used by LLMFactory.create_rate_limited() to wrap single-model providers
    with LLMRateLimiter. GroqModelPool does NOT use this — it uses the tracker
    instead for reactive header-based rate management.

    Falls back to conservative defaults and logs a warning when the model is
    not found, so unknown models degrade gracefully rather than flooding 429s.

    Args:
        model_name: Exact model ID string e.g. 'gemini-2.5-flash'.
        max_concurrent: Max simultaneous in-flight requests (semaphore size).
        burst_multiplier: Burst tolerance multiplier above the sustained RPM rate.

    Returns:
        RateLimiterConfig ready to pass to LLMRateLimiter.
    """
    limits = get_model_limits(model_name)

    logger.debug(
        "Rate limits for model '%s': rpm=%d, rpd=%d, tpm=%d, tpd=%d",
        model_name, limits.rpm, limits.rpd, limits.tpm, limits.tpd,
    )

    return RateLimiterConfig(
        rpm=limits.rpm,
        rpd=limits.rpd,
        max_concurrent=max_concurrent,
        burst_multiplier=burst_multiplier,
    )
