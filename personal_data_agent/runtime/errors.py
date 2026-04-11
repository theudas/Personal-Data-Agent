class AgentError(Exception):
    """Base exception for this project."""


class ValidationError(AgentError):
    """Raised when tool arguments do not satisfy schema."""


class SecurityError(AgentError):
    """Raised when path escapes notes root or file type is unsupported."""


class ToolExecutionError(AgentError):
    """Raised when a tool fails during execution."""

    def __init__(self, message: str, retriable: bool = False):
        super().__init__(message)
        self.retriable = retriable
