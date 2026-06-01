"""Per-model rate limit registry; limits sourced from official provider docs and not tunable via settings."""

from dataclasses import dataclass
from typing import Literal, Optional

from llm.rate_limiter.rate_limiter_config import RateLimiterConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# conservative fallback so unknown models degrade gracefully
_UNKNOWN_MODEL_RPM = 10
_UNKNOWN_MODEL_RPD = 200
_UNKNOWN_MODEL_TPM = 10_000
_UNKNOWN_MODEL_TPD = 100_000

# pool membership literals used by ModelRouter
PoolTag = Literal["FAST", "STRONG", "BOTH"]


@dataclass(frozen=True)
class _ModelLimits:
    """Provider-enforced hard limits for one model across rpm, rpd, tpm, tpd dimensions."""

    rpm: int
    rpd: int
    tpm: int
    tpd: int
    pool: Optional[PoolTag] = None


# registry — only active models

MODEL_RATE_LIMITS: dict[str, _ModelLimits] = {

    # groq — fast pool
    "llama-3.1-8b-instant":                      _ModelLimits(rpm=30,  rpd=14_400, tpm=6_000,  tpd=500_000, pool="FAST"),
    "openai/gpt-oss-20b":                        _ModelLimits(rpm=30,  rpd=1_000,  tpm=8_000,  tpd=200_000, pool="FAST"),

    # groq — strong pool
    "llama-3.3-70b-versatile":                   _ModelLimits(rpm=30,  rpd=1_000,  tpm=12_000, tpd=100_000, pool="STRONG"),
    "meta-llama/llama-4-scout-17b-16e-instruct": _ModelLimits(rpm=30,  rpd=1_000,  tpm=30_000, tpd=500_000, pool="STRONG"),

    # groq — both pools (shared budget)
    "qwen/qwen3-32b":                            _ModelLimits(rpm=60,  rpd=1_000,  tpm=6_000,  tpd=500_000, pool="BOTH"),

    # gemini — fallback provider
    "gemini-2.5-flash":                          _ModelLimits(rpm=10,  rpd=500,    tpm=250_000, tpd=1_000_000),
}


def get_model_limits(model_name: str) -> _ModelLimits:
    """Return the _ModelLimits for the given model, with a conservative fallback if not registered."""
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
    """Return a RateLimiterConfig with the correct RPM/RPD for the given model."""
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
