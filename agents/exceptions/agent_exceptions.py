"""Agent-layer exceptions; inner-layer LLM errors propagate as-is."""


class AgentError(Exception):
    """Base exception for all agent-layer errors."""

    def __init__(self, message: str, details: dict = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(message)


class AgentPlanningError(AgentError):
    """Raised when query decomposition fails."""


class AgentRetrievalError(AgentError):
    """Raised when sub-query execution fails completely."""


class AgentSynthesisError(AgentError):
    """Raised when final answer synthesis fails."""
