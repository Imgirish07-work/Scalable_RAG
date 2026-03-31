"""Agent-layer exceptions.

Agent errors wrap failures in planning, sub-query execution, verification,
and synthesis. LLM-layer errors (LLMAuthError, LLMRateLimitError) propagate
as-is — agents do NOT blanket-wrap inner-layer exceptions.
"""


class AgentError(Exception):
    """Base exception for all agent-layer errors.

    Attributes:
        message: Human-readable error description.
        details: Structured dict for logging and debugging.
    """

    def __init__(self, message: str, details: dict = None) -> None:
        """Initialize AgentError.

        Args:
            message: Human-readable error description.
            details: Optional structured context for logging.
        """
        self.message = message
        self.details = details or {}
        super().__init__(message)


class AgentPlanningError(AgentError):
    """Raised when query decomposition fails.

    Covers LLM planning call failure, unparseable plan output,
    or zero sub-queries produced.
    """


class AgentRetrievalError(AgentError):
    """Raised when sub-query execution fails.

    Covers complete failure of all sub-queries. Partial failures
    (some sub-queries succeed, some fail) are handled gracefully
    by the verifier — this exception is for total failure only.
    """


class AgentSynthesisError(AgentError):
    """Raised when final answer synthesis fails.

    Covers LLM synthesis call failure or empty synthesis output.
    """
