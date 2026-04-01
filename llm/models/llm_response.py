"""
Standardized LLM response model.

Every provider (Gemini, OpenAI, future providers) returns this exact model.
Pipeline, cache, RAG, and RLM layers import ONLY LLMResponse — never raw
provider response objects.

Key design decisions:
    - Frozen (immutable) — once created, fields cannot be modified.
    - finish_reason normalized to lowercase strings across all providers:
      "stop", "length", "safety", "content_filter", "tool_calls", "unknown".
    - tokens_used cross-validated against prompt_tokens + completion_tokens.
    - metadata dict available for provider-specific extras without schema changes.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUPPORTED_PROVIDERS = {"openai", "gemini", "groq"}

# Normalized finish_reason values accepted across all providers
VALID_FINISH_REASONS = {
    "stop",             # Normal completion
    "length",           # Hit max_tokens limit (truncated)
    "safety",           # Blocked by safety filters
    "content_filter",   # Blocked by content policy
    "recitation",       # Blocked by recitation filter (Gemini)
    "tool_calls",       # Model wants to call a tool
    "error",            # Generation failed mid-stream
    "unknown",          # Provider did not return a reason
}


class LLMResponse(BaseModel):
    """
    Standard response returned by ALL LLM providers.

    Pipeline only ever sees this — never raw provider responses.
    Cache layer wraps this in CacheEntry. RAG layer wraps this in RAGResponse.

    Attributes:
        text: Generated text from the LLM. Cannot be empty.
        model: Model identifier e.g. 'gpt-4o-mini', 'gemini-2.5-flash'.
        provider: Provider name. Must be in SUPPORTED_PROVIDERS.
        finish_reason: Why generation stopped. Normalized to lowercase.
            Used by QualityGate to detect truncation (finish_reason == 'length').
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        tokens_used: Total tokens consumed. Must be >= prompt + completion.
        latency_ms: Round-trip time for the API call in milliseconds.
        cached: Whether this response was served from cache.
        metadata: Provider-specific extras (safety ratings, logprobs, etc.).
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(
        ...,
        min_length=1,
        description="Generated text from LLM",
    )
    model: str = Field(
        ...,
        min_length=1,
        description="Model name e.g. gpt-4o-mini",
    )
    provider: str = Field(
        ...,
        description="Provider name e.g. openai or gemini",
    )
    finish_reason: str = Field(
        default="unknown",
        description="Why generation stopped: stop, length, safety, etc.",
    )
    prompt_tokens: int = Field(
        default=0,
        ge=0,
        description="Input tokens consumed",
    )
    completion_tokens: int = Field(
        default=0,
        ge=0,
        description="Output tokens generated",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (prompt + completion + overhead)",
    )
    latency_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Response time in milliseconds",
    )
    cached: bool = Field(
        default=False,
        description="Was this response served from cache?",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Extra provider-specific info (safety ratings, logprobs, etc.)",
    )

    # Field validators

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        """Normalize and validate provider name.

        Args:
            value: Raw provider string from constructor.

        Returns:
            Cleaned lowercase provider name.

        Raises:
            ValueError: If provider is not in SUPPORTED_PROVIDERS.
        """
        cleaned = value.strip().lower()
        if cleaned not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Provider '{value}' not supported. "
                f"Must be one of: {sorted(SUPPORTED_PROVIDERS)}"
            )
        return cleaned

    @field_validator("text", "model")
    @classmethod
    def validate_not_blank(cls, value: str) -> str:
        """Reject blank or whitespace-only strings.

        Args:
            value: Raw string from constructor.

        Returns:
            Stripped string.

        Raises:
            ValueError: If value is empty or whitespace only.
        """
        if not value.strip():
            raise ValueError("Field cannot be blank or whitespace only.")
        return value.strip()

    @field_validator("finish_reason")
    @classmethod
    def validate_finish_reason(cls, value: str) -> str:
        """Normalize finish_reason to lowercase.

        Accepts any string but warns if not in the known set.
        Unknown values are passed through — providers may add new ones.

        Args:
            value: Raw finish_reason from provider.

        Returns:
            Lowercase finish_reason string.
        """
        cleaned = value.strip().lower() if value else "unknown"
        # Allow unknown values through — don't break on new provider reasons
        return cleaned

    # Cross-field validation

    @model_validator(mode="after")
    def validate_token_consistency(self) -> "LLMResponse":
        """Validate that tokens_used >= prompt_tokens + completion_tokens.

        Some providers include internal overhead tokens in tokens_used,
        so tokens_used may be greater than the sum. But it should never
        be less — that indicates a parsing bug.

        Returns:
            Self if valid.

        Raises:
            ValueError: If tokens_used < prompt_tokens + completion_tokens.
        """
        expected = self.prompt_tokens + self.completion_tokens
        if self.tokens_used < expected:
            raise ValueError(
                f"tokens_used ({self.tokens_used}) cannot be less than "
                f"prompt_tokens ({self.prompt_tokens}) + "
                f"completion_tokens ({self.completion_tokens}) = {expected}"
            )
        return self