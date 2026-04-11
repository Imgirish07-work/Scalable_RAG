"""
Standardized response model returned by all LLM providers.

Design:
    Frozen Pydantic model (immutable after construction). Every provider returns
    this exact model — the pipeline never sees raw provider response objects.
    finish_reason is normalized to a consistent lowercase string across all
    providers. tokens_used is cross-validated against prompt + completion counts.

Chain of Responsibility:
    Provider._parse_response() constructs LLMResponse → returned to
    BaseRAG.generate() → optionally wrapped in CacheEntry by the cache layer
    → wrapped in RAGResponse by the RAG layer.

Dependencies:
    pydantic (BaseModel, ConfigDict, Field, field_validator, model_validator).
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUPPORTED_PROVIDERS = {"openai", "gemini", "groq"}

# Normalized finish_reason strings accepted from all providers
VALID_FINISH_REASONS = {
    "stop",             # Normal completion
    "length",           # Truncated at max_tokens limit
    "safety",           # Blocked by safety filters
    "content_filter",   # Blocked by content policy
    "recitation",       # Blocked by recitation filter (Gemini-specific)
    "tool_calls",       # Model requested a tool call
    "error",            # Generation failed mid-stream
    "unknown",          # Provider did not return a reason
}


class LLMResponse(BaseModel):
    """Standard response returned by all LLM providers.

    The pipeline only ever sees this model — never raw provider responses.
    The cache layer wraps this in CacheEntry; the RAG layer wraps it in
    RAGResponse.

    Attributes:
        text: Generated text. Cannot be empty or whitespace-only.
        model: Model identifier e.g. 'gpt-4o-mini', 'gemini-2.5-flash'.
        provider: Provider name. Must be in SUPPORTED_PROVIDERS.
        finish_reason: Why generation stopped. Normalized to lowercase.
            QualityGate uses 'length' to detect truncated responses.
        prompt_tokens: Number of input tokens consumed.
        completion_tokens: Number of output tokens generated.
        tokens_used: Total tokens consumed. Must be >= prompt + completion.
        latency_ms: Round-trip time for the API call in milliseconds.
        cached: True if this response was served from cache.
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
        """Normalize and validate the provider name.

        Args:
            value: Raw provider string from the constructor.

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
            value: Raw string from the constructor.

        Returns:
            Stripped string.

        Raises:
            ValueError: If value is empty or whitespace-only.
        """
        if not value.strip():
            raise ValueError("Field cannot be blank or whitespace only.")
        return value.strip()

    @field_validator("finish_reason")
    @classmethod
    def validate_finish_reason(cls, value: str) -> str:
        """Normalize finish_reason to lowercase.

        Unknown values are passed through rather than rejected — providers
        may introduce new finish reasons in future API versions.

        Args:
            value: Raw finish_reason string from the provider.

        Returns:
            Lowercase finish_reason string.
        """
        cleaned = value.strip().lower() if value else "unknown"
        # Pass unknown values through to avoid breaking on new provider reasons
        return cleaned

    # Cross-field validation

    @model_validator(mode="after")
    def validate_token_consistency(self) -> "LLMResponse":
        """Validate that tokens_used is not less than prompt + completion tokens.

        Some providers include internal overhead in tokens_used, so it may
        legitimately exceed the sum. But it should never be less — that
        indicates a parsing or provider bug.

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
