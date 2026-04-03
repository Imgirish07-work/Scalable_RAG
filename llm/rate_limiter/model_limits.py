"""
Model-based rate limit registry.

Single source of truth for per-model RPM and RPD limits.
Covers free-tier limits for Groq, Gemini, and OpenAI.

Usage:
    config = get_rate_limit_config("llama-3.3-70b-versatile")
    limiter = LLMRateLimiter(provider, config)

Adding a new model:
    Add one line to MODEL_RATE_LIMITS with the exact model ID string
    and its real RPM/RPD from the provider's API docs.

Why not env vars?
    RPM/RPD are PROVIDER-ENFORCED CONSTANTS — they don't change based
    on deployment config. Putting them in settings creates the illusion
    they are tunable, which causes 429s when someone sets rpm=60 for
    a Groq model that allows only 30.
"""

from dataclasses import dataclass

from llm.rate_limiter.rate_limiter_config import RateLimiterConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Conservative defaults used when a model is not in the registry.
# Deliberately low so unknown models degrade gracefully rather than 429.
_UNKNOWN_MODEL_RPM = 10
_UNKNOWN_MODEL_RPD = 200


@dataclass(frozen=True)
class _ModelLimits:
    rpm: int   # requests per minute
    rpd: int   # requests per day


# Registry

MODEL_RATE_LIMITS: dict[str, _ModelLimits] = {

    # Groq free tier
    "llama-3.1-8b-instant":             _ModelLimits(rpm=30, rpd=14_400),
    "llama-3.3-70b-versatile":          _ModelLimits(rpm=30, rpd=1_000),
    "llama-3.2-1b-preview":             _ModelLimits(rpm=30, rpd=14_400),
    "llama-3.2-3b-preview":             _ModelLimits(rpm=30, rpd=14_400),
    "llama-3.2-11b-vision-preview":     _ModelLimits(rpm=15, rpd=3_500),
    "llama-3.2-90b-vision-preview":     _ModelLimits(rpm=15, rpd=3_500),
    "llama-3.1-70b-versatile":          _ModelLimits(rpm=30, rpd=14_400),
    "mixtral-8x7b-32768":               _ModelLimits(rpm=30, rpd=14_400),
    "gemma2-9b-it":                     _ModelLimits(rpm=30, rpd=14_400),
    "gemma-7b-it":                      _ModelLimits(rpm=30, rpd=14_400),
    "qwen/qwen3-32b":                   _ModelLimits(rpm=30, rpd=1_000),
    "qwen-qwq-32b":                     _ModelLimits(rpm=30, rpd=1_000),
    "deepseek-r1-distill-llama-70b":    _ModelLimits(rpm=30, rpd=1_000),

    # Gemini free tier
    "gemini-2.5-flash":                 _ModelLimits(rpm=10,  rpd=500),
    "gemini-2.5-flash-preview-04-17":   _ModelLimits(rpm=10,  rpd=500),
    "gemini-2.0-flash":                 _ModelLimits(rpm=15,  rpd=1_500),
    "gemini-2.0-flash-lite":            _ModelLimits(rpm=30,  rpd=1_500),
    "gemini-1.5-flash":                 _ModelLimits(rpm=15,  rpd=1_500),
    "gemini-1.5-flash-8b":              _ModelLimits(rpm=15,  rpd=1_500),
    "gemini-1.5-pro":                   _ModelLimits(rpm=2,   rpd=50),
    "gemini-1.0-pro":                   _ModelLimits(rpm=15,  rpd=1_500),

    # OpenAI free
    "gpt-3.5-turbo":                    _ModelLimits(rpm=3,   rpd=200),
    "gpt-3.5-turbo-instruct":           _ModelLimits(rpm=3,   rpd=200),
    "gpt-4o-mini":                      _ModelLimits(rpm=3,   rpd=200),
    "gpt-4o":                           _ModelLimits(rpm=3,   rpd=200),
    "gpt-4":                            _ModelLimits(rpm=3,   rpd=200),
    "gpt-4-turbo":                      _ModelLimits(rpm=3,   rpd=200),
}


def get_rate_limit_config(
    model_name: str,
    max_concurrent: int = 5,
    burst_multiplier: float = 1.0,
) -> RateLimiterConfig:
    """Return a RateLimiterConfig with the correct RPM/RPD for a given model.

    Looks up the model in MODEL_RATE_LIMITS. Falls back to conservative
    defaults and logs a warning when the model is not found, so unknown
    models degrade gracefully rather than causing 429 floods.

    Args:
        model_name: Exact model ID string (e.g. 'llama-3.3-70b-versatile').
        max_concurrent: Max simultaneous in-flight requests (semaphore size).
        burst_multiplier: Burst tolerance above sustained RPM rate.

    Returns:
        RateLimiterConfig ready to pass to LLMRateLimiter.
    """
    limits = MODEL_RATE_LIMITS.get(model_name)

    if limits is None:
        logger.warning(
            "No rate limits registered for model '%s'. "
            "Using conservative fallback (rpm=%d, rpd=%d). "
            "Add this model to MODEL_RATE_LIMITS in "
            "llm/rate_limiter/model_limits.py to silence this warning.",
            model_name,
            _UNKNOWN_MODEL_RPM,
            _UNKNOWN_MODEL_RPD,
        )
        limits = _ModelLimits(rpm=_UNKNOWN_MODEL_RPM, rpd=_UNKNOWN_MODEL_RPD)
    else:
        logger.debug(
            "Rate limits for model '%s': rpm=%d, rpd=%d",
            model_name, limits.rpm, limits.rpd,
        )

    return RateLimiterConfig(
        rpm=limits.rpm,
        rpd=limits.rpd,
        max_concurrent=max_concurrent,
        burst_multiplier=burst_multiplier,
    )
