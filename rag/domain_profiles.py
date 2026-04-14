"""
Domain profile registry — per-request RAG parameter overrides.

Registry pattern: profiles registered once at module import, looked up per request.
Caller-supplied RAGConfig values always win over profile defaults.
"""

from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DomainProfile:
    # Immutable parameter set for a named domain.
    top_k: int
    max_context_tokens: int
    min_context_chunks: int
    reranker_score_threshold: float
    temperature: float


class DomainRegistry:
    # Class-level dict — shared across all instances, populated at import time.
    _profiles: dict[str, DomainProfile] = {}

    @classmethod
    def register(cls, name: str, profile: DomainProfile) -> None:
        # Register or overwrite a named profile.
        cls._profiles[name] = profile

    @classmethod
    def get(cls, name: str) -> DomainProfile | None:
        # Return the profile for name, or None if not registered.
        return cls._profiles.get(name)

    @classmethod
    def names(cls) -> list[str]:
        # Return all registered profile names.
        return list(cls._profiles)


# --- Built-in profiles ---

# technical: factual / Q&A — tight context window, high precision threshold.
DomainRegistry.register("technical", DomainProfile(
    top_k=5,
    max_context_tokens=3072,
    min_context_chunks=2,
    reranker_score_threshold=0.12,
    temperature=0.1,
))

# story: narrative / creative — broad retrieval, relaxed scoring, rich context.
DomainRegistry.register("story", DomainProfile(
    top_k=12,
    max_context_tokens=6144,
    min_context_chunks=4,
    reranker_score_threshold=0.05,
    temperature=0.3,
))


def apply_domain_profile(config_kwargs: dict, domain: str | None) -> dict:
    # Merge profile defaults into config_kwargs.
    # Profile supplies defaults — keys already in config_kwargs are NOT overwritten.
    # Returns config_kwargs unchanged when domain is None or unregistered.
    if not domain:
        return config_kwargs

    profile = DomainRegistry.get(domain)
    if profile is None:
        logger.warning("DomainProfile | unknown domain | name=%s", domain)
        return config_kwargs

    # Profile = floor values; caller wins on any key already set.
    merged = {
        "top_k": profile.top_k,
        "max_context_tokens": profile.max_context_tokens,
        "min_context_chunks": profile.min_context_chunks,
        "reranker_score_threshold": profile.reranker_score_threshold,
        "temperature": profile.temperature,
    }
    merged.update(config_kwargs)  # caller overrides win

    logger.info(
        "DomainProfile | applied | domain=%s | top_k=%d | threshold=%.2f | temp=%.1f",
        domain,
        merged["top_k"],
        merged["reranker_score_threshold"],
        merged["temperature"],
    )

    return merged
