"""
Per-model rate limit registry for the LLM rate limiter.

Design:
    Module-level registry dict (MODEL_RATE_LIMITS) maps exact model ID strings
    to frozen _ModelLimits dataclasses. get_rate_limit_config() looks up the
    model and returns a RateLimiterConfig ready for LLMRateLimiter.

    RPM, RPD, TPM, and TPD values are provider-enforced constants, not
    deployment config. Storing them here (rather than in settings/.env) prevents
    false impressions that they are tunable — setting rpm=60 for a model
    limited to 30 causes 429s.

    pool: Marks which Groq model pool(s) a model belongs to:
        "FAST"   — lightweight tasks (eval, rewrite, classify) max_tokens≤512
        "STRONG" — final synthesis / complex generation  max_tokens>512 or None
        "BOTH"   — shared budget; model appears in both pools (e.g. qwen3-32b)
        None     — non-Groq provider; pool routing does not apply

    Adding a new Groq model requires exactly one line in MODEL_RATE_LIMITS.

Chain of Responsibility:
    LLMFactory.create_rate_limited() calls get_rate_limit_config(model_name) →
    returns RateLimiterConfig → passed to LLMRateLimiter constructor.

    GroqModelPool reads MODEL_RATE_LIMITS directly via get_model_limits() to
    build its per-model RateLimitTracker budget.

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

    rpm: int                     # requests per minute
    rpd: int                     # requests per day
    tpm: int                     # tokens per minute
    tpd: int                     # tokens per day
    pool: Optional[PoolTag] = None  # Groq pool tag; None = not a Groq pool model


# Registry — add ONE line per new Groq model using the exact model ID

MODEL_RATE_LIMITS: dict[str, _ModelLimits] = {

    # Groq free tier — Groq Model Pool members
    # Priority order within pools is defined in ModelRouter, not here.

    # FAST pool — lightweight tasks, high volume, sub-512-token responses
    "llama-3.1-8b-instant":             _ModelLimits(rpm=30, rpd=14_400, tpm=20_000,  tpd=500_000,  pool="FAST"),

    # STRONG pool — final synthesis, complex reasoning, large output
    "moonshotai/kimi-k2":               _ModelLimits(rpm=30, rpd=400,    tpm=12_800,  tpd=300_000,  pool="STRONG"),
    "llama-3.3-70b-versatile":          _ModelLimits(rpm=30, rpd=1_000,  tpm=12_000,  tpd=100_000,  pool="STRONG"),
    "meta-llama/llama-4-scout-17b-16e-instruct": _ModelLimits(rpm=30, rpd=400, tpm=12_800, tpd=500_000, pool="STRONG"),

    # BOTH pools — shared budget; one RateLimitState instance, referenced by both pools
    "qwen/qwen3-32b":                   _ModelLimits(rpm=30, rpd=400,    tpm=12_800,  tpd=500_000,  pool="BOTH"),

    # Groq free tier — legacy / non-pool models (kept for backward compat)
    "llama-3.2-1b-preview":             _ModelLimits(rpm=30, rpd=14_400, tpm=7_000,   tpd=500_000),
    "llama-3.2-3b-preview":             _ModelLimits(rpm=30, rpd=14_400, tpm=7_000,   tpd=500_000),
    "llama-3.2-11b-vision-preview":     _ModelLimits(rpm=15, rpd=3_500,  tpm=7_000,   tpd=250_000),
    "llama-3.2-90b-vision-preview":     _ModelLimits(rpm=15, rpd=3_500,  tpm=7_000,   tpd=250_000),
    "llama-3.1-70b-versatile":          _ModelLimits(rpm=30, rpd=14_400, tpm=20_000,  tpd=500_000),
    "mixtral-8x7b-32768":               _ModelLimits(rpm=30, rpd=14_400, tpm=5_000,   tpd=500_000),
    "gemma2-9b-it":                     _ModelLimits(rpm=30, rpd=14_400, tpm=15_000,  tpd=500_000),
    "gemma-7b-it":                      _ModelLimits(rpm=30, rpd=14_400, tpm=15_000,  tpd=500_000),
    "qwen-qwq-32b":                     _ModelLimits(rpm=30, rpd=1_000,  tpm=6_000,   tpd=500_000),
    "deepseek-r1-distill-llama-70b":    _ModelLimits(rpm=30, rpd=1_000,  tpm=6_000,   tpd=500_000),

    # Gemini free tier
    "gemini-2.5-flash":                 _ModelLimits(rpm=10,  rpd=500,   tpm=250_000, tpd=1_000_000),
    "gemini-2.5-flash-preview-04-17":   _ModelLimits(rpm=10,  rpd=500,   tpm=250_000, tpd=1_000_000),
    "gemini-2.0-flash":                 _ModelLimits(rpm=15,  rpd=1_500, tpm=1_000_000, tpd=1_000_000),
    "gemini-2.0-flash-lite":            _ModelLimits(rpm=30,  rpd=1_500, tpm=1_000_000, tpd=1_000_000),
    "gemini-1.5-flash":                 _ModelLimits(rpm=15,  rpd=1_500, tpm=1_000_000, tpd=1_000_000),
    "gemini-1.5-flash-8b":              _ModelLimits(rpm=15,  rpd=1_500, tpm=1_000_000, tpd=1_000_000),
    "gemini-1.5-pro":                   _ModelLimits(rpm=2,   rpd=50,    tpm=32_000,  tpd=50_000),
    "gemini-1.0-pro":                   _ModelLimits(rpm=15,  rpd=1_500, tpm=32_000,  tpd=1_000_000),

    # OpenAI free tier
    "gpt-3.5-turbo":                    _ModelLimits(rpm=3,   rpd=200,   tpm=40_000,  tpd=200_000),
    "gpt-3.5-turbo-instruct":           _ModelLimits(rpm=3,   rpd=200,   tpm=40_000,  tpd=200_000),
    "gpt-4o-mini":                      _ModelLimits(rpm=3,   rpd=200,   tpm=40_000,  tpd=200_000),
    "gpt-4o":                           _ModelLimits(rpm=3,   rpd=200,   tpm=10_000,  tpd=100_000),
    "gpt-4":                            _ModelLimits(rpm=3,   rpd=200,   tpm=10_000,  tpd=100_000),
    "gpt-4-turbo":                      _ModelLimits(rpm=3,   rpd=200,   tpm=10_000,  tpd=100_000),
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
        model_name: Exact model ID string e.g. 'llama-3.3-70b-versatile'.
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
