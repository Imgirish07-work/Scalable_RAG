"""
Custom exception hierarchy for LLM provider errors.

Design:
    Every provider translates its SDK-specific exceptions into this hierarchy
    before raising. The pipeline and RAG layers catch only these exceptions,
    never raw OpenAI or Gemini SDK errors. LLMError is the base class —
    catching it handles any provider failure uniformly.

Chain of Responsibility:
    Provider._handle_error() translates SDK exceptions → raises LLMError
    subclass → caught by BaseRAG.generate() or LLMRateLimiter.

Dependencies:
    None (stdlib only).
"""


class LLMError(Exception):
    """Base exception for all LLM-related errors.

    Catch this to handle any provider failure uniformly. All provider-specific
    errors are subclasses of LLMError so a single except clause is sufficient
    when the caller does not need to distinguish failure modes.
    """


class LLMAuthError(LLMError):
    """API key is invalid or missing.

    Raised when:
        - API key is not set in .env or passed to the constructor.
        - The provider returns a 401 or 403 (or equivalent) response.
    """


class LLMRateLimitError(LLMError):
    """API quota or rate limit has been exceeded.

    Raised when:
        - The provider returns HTTP 429 or ResourceExhausted.
        - Caller should apply exponential backoff before retrying.
    """


class LLMTimeoutError(LLMError):
    """Request exceeded the configured deadline.

    Raised when:
        - The provider returns DeadlineExceeded or APITimeoutError.
        - Consider increasing the timeout or reducing the prompt size.
    """


class LLMTokenLimitError(LLMError):
    """Prompt exceeds the model's context window.

    Raised when:
        - The provider returns context_length_exceeded or equivalent.
        - The RLM layer uses this as a signal to chunk and recurse.
    """


class LLMProviderError(LLMError):
    """Generic provider-side error not covered by a more specific subclass.

    Raised when:
        - The provider returns an unrecognized API error.
        - Unknown SDK exceptions are wrapped in this class.
    """
