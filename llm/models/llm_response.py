"""Standardized frozen Pydantic response model returned by all LLM providers."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SUPPORTED_PROVIDERS = {"openai", "gemini", "groq"}

# normalized finish_reason strings accepted from all providers
VALID_FINISH_REASONS = {
    "stop",
    "length",
    "safety",
    "content_filter",
    "recitation",
    "tool_calls",
    "error",
    "unknown",
}


class LLMResponse(BaseModel):
    """Standard frozen response model returned by all LLM providers."""

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

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        """Normalize and validate the provider name."""
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
        """Reject blank or whitespace-only strings."""
        if not value.strip():
            raise ValueError("Field cannot be blank or whitespace only.")
        return value.strip()

    @field_validator("finish_reason")
    @classmethod
    def validate_finish_reason(cls, value: str) -> str:
        """Normalize finish_reason to lowercase; unknown values pass through."""
        cleaned = value.strip().lower() if value else "unknown"
        return cleaned

    @model_validator(mode="after")
    def validate_token_consistency(self) -> "LLMResponse":
        """Validate that tokens_used >= prompt + completion tokens."""
        expected = self.prompt_tokens + self.completion_tokens
        if self.tokens_used < expected:
            raise ValueError(
                f"tokens_used ({self.tokens_used}) cannot be less than "
                f"prompt_tokens ({self.prompt_tokens}) + "
                f"completion_tokens ({self.completion_tokens}) = {expected}"
            )
        return self
