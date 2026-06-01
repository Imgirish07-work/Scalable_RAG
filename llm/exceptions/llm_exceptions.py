"""Custom exception hierarchy for LLM provider errors; providers translate SDK exceptions into these."""


class LLMError(Exception):
    """Base exception for all LLM-related errors."""


class LLMAuthError(LLMError):
    """API key is invalid or missing."""


class LLMRateLimitError(LLMError):
    """API quota or rate limit has been exceeded."""


class LLMTimeoutError(LLMError):
    """Request exceeded the configured deadline."""


class LLMTokenLimitError(LLMError):
    """Prompt exceeds the model's context window."""


class LLMProviderError(LLMError):
    """Generic provider-side error not covered by a more specific subclass."""
