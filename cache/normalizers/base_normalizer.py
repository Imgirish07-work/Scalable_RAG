"""
Abstract base class for query normalization steps.

Each normalizer is a single transformation in the Chain of Responsibility.
They are composed by QueryNormalizerChain which runs them in sequence.

Design:
    - Each step does ONE thing (SRP)
    - Each step is independently testable
    - Steps are sync (CPU only — Rule 2)
    - Steps never raise — they return input unchanged on failure
    - Steps are stateless — no instance variables mutated between calls

Chain of Responsibility pattern:
    raw_query → Step1.normalize() → Step2.normalize() → ... → normalized_query

Adding a new step:
    1. Subclass BaseNormalizer
    2. Implement normalize()
    3. Add to QueryNormalizerChain._build_chain()
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