"""Domain profile registry; caller-supplied RAGConfig values always win over profile defaults."""

from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DomainProfile:
    top_k: int
    max_context_tokens: int
    min_context_chunks: int
    reranker_score_threshold: float
    temperature: float


class DomainRegistry:
    _profiles: dict[str, DomainProfile] = {}

    @classmethod
    def register(cls, name: str, profile: DomainProfile) -> None:
        cls._profiles[name] = profile

    @classmethod
    def get(cls, name: str) -> DomainProfile | None:
        return cls._profiles.get(name)

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._profiles)


DomainRegistry.register("technical", DomainProfile(
    top_k=5,
    max_context_tokens=3072,
    min_context_chunks=2,
    reranker_score_threshold=0.12,
    temperature=0.1,
))

DomainRegistry.register("story", DomainProfile(
    top_k=12,
    max_context_tokens=6144,
    min_context_chunks=4,
    reranker_score_threshold=0.05,
    temperature=0.3,
))


def apply_domain_profile(config_kwargs: dict, domain: str | None) -> dict:
    """Merge profile defaults into config_kwargs; caller-set keys are not overwritten."""
    if not domain:
        return config_kwargs

    profile = DomainRegistry.get(domain)
    if profile is None:
        logger.warning("DomainProfile | unknown domain | name=%s", domain)
        return config_kwargs

    merged = {
        "top_k": profile.top_k,
        "max_context_tokens": profile.max_context_tokens,
        "min_context_chunks": profile.min_context_chunks,
        "reranker_score_threshold": profile.reranker_score_threshold,
        "temperature": profile.temperature,
    }
    merged.update(config_kwargs)

    logger.info(
        "DomainProfile | applied | domain=%s | top_k=%d | threshold=%.2f | temp=%.1f",
        domain,
        merged["top_k"],
        merged["reranker_score_threshold"],
        merged["temperature"],
    )

    return merged
