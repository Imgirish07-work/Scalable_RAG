"""
Per-model rate limit registry for the LLM rate limiter.

Design:
    Module-level registry dict (MODEL_RATE_LIMITS) maps exact model ID strings
    to frozen _ModelLimits dataclasses. get_rate_limit_config() looks up the
    model and returns a RateLimiterConfig ready for LLMRateLimiter.

    RPM and RPD values are provider-enforced constants, not deployment config.
    Storing them here (rather than in settings/.env) prevents false impressions
    that they are tunable — setting rpm=60 for a model limited to 30 causes 429s.

Chain of Responsibility:
    LLMFactory.create_rate_limited() calls get_rate_limit_config(model_name) →
    returns RateLimiterConfig → passed to LLMRateLimiter constructor.

Dependencies:
    llm.rate_limiter.rate_limiter_config.RateLimiterConfig, utils.logger.
"""

from dataclasses import dataclass

from llm.rate_limiter.rate_limiter_config import RateLimiterConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Conservative fallback limits used for models not found in the registry.
# Deliberately low so unknown models degrade gracefully rather than causing 429 floods.
_UNKNOWN_MODEL_RPM = 10
_UNKNOWN_MODEL_RPD = 200


@dataclass(frozen=True)
class _ModelLimits:
    rpm: int   # requests per minute
    rpd: int   # requests per day


# Registry — add one line per new model using the exact model ID from provider docs

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

    # OpenAI free tier
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
    """Return a RateLimiterConfig with the correct RPM/RPD for the given model.

    Falls back to conservative defaults and logs a warning when the model is
    not found, so unknown models degrade gracefully rather than flooding 429s.

    Args:
        model_name: Exact model ID string e.g. 'llama-3.3-70b-versatile'.
        max_concurrent: Max simultaneous in-flight requests (semaphore size).
        burst_multiplier: Burst tolerance multiplier above the sustained RPM rate.

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
