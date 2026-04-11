"""
Abstract base class for query normalization steps.

Design:
    Each normalizer is a single transformation in a Chain of Responsibility
    pipeline. Steps are composed by QueryNormalizerChain, which runs them
    in sequence. Each step follows SRP: it does exactly one thing and is
    independently testable.

    Step contract:
        - normalize() is sync (CPU only — no I/O)
        - Steps never raise — return input unchanged on failure
        - Steps are stateless — no instance variables mutated between calls

    Adding a new step:
        1. Subclass BaseNormalizer
        2. Implement normalize()
        3. Add to QueryNormalizerChain._build_default_chain()

Chain of Responsibility:
    raw_query → Step1.normalize() → Step2.normalize() → ... → normalized_query
    Instantiated steps are composed by QueryNormalizerChain.

Dependencies:
    abc (stdlib only)
"""

from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    """Single normalization step in the query preprocessing chain."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Step name for logging and debugging."""
        ...

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Apply this normalization step to the input text.

        Args:
            text: Input text (may already be partially normalized
                  by previous steps in the chain).

        Returns:
            Normalized text. Must return input unchanged if the step
            is not applicable or encounters an error.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
