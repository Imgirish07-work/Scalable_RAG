"""
LLM exception hierarchy.

Design:
    - Every provider translates SDK-specific errors into this hierarchy.
    - Pipeline and RAG layers catch ONLY these exceptions — never raw
      OpenAI/Gemini SDK errors.
    - LLMError is the base — catching it catches everything.

Hierarchy:
    LLMError
    ├── LLMAuthError          (invalid / missing API key)
    ├── LLMRateLimitError     (quota exceeded — retry after delay)
    ├── LLMTimeoutError       (request exceeded deadline)
    ├── LLMTokenLimitError    (prompt exceeds context window)
    └── LLMProviderError      (generic provider-side failure)
"""


class LLMError(Exception):
    """Base exception for all LLM errors.

    All provider-specific errors are translated into subclasses of this.
    Pipeline code catches LLMError to handle any provider failure uniformly.
    """


class LLMAuthError(LLMError):
    """Invalid or missing API key.

    Raised when:
        - API key is not set in .env or constructor.
        - Provider returns 401/403 or equivalent authentication failure.
    """


class LLMRateLimitError(LLMError):
    """API quota or rate limit exceeded.

    Raised when:
        - Provider returns 429 or ResourceExhausted.
        - Caller should implement exponential backoff before retrying.
    """


class LLMTimeoutError(LLMError):
    """Request exceeded the configured deadline.

    Raised when:
        - Provider returns DeadlineExceeded or APITimeoutError.
        - Consider increasing timeout or reducing prompt size.
    """


class LLMTokenLimitError(LLMError):
    """Prompt exceeds the model's context window.

    Raised when:
        - Provider returns context_length_exceeded or equivalent.
        - RLM layer uses this as a signal to chunk and recurse.
    """


class LLMProviderError(LLMError):
    """Generic provider-side error not covered by specific subclasses.

    Raised when:
        - Provider returns an unrecognized API error.
        - Unknown exceptions from the SDK are wrapped in this.
    """