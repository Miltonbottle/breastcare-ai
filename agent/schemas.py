"""Serializable data structures for BreastCareAgent execution records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentTraceStep:
    """One bounded tool invocation in the agent workflow."""

    tool: str
    status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {"tool": self.tool, "status": self.status}
        if self.details:
            result["details"] = self.details
        return result


class AgentWorkflowError(Exception):
    """Controlled workflow failure carrying the audit trail completed so far."""

    def __init__(self, message: str, status_code: int, trace: list[AgentTraceStep]):
        super().__init__(message)
        self.status_code = status_code
        self.trace = trace

    def trace_dict(self) -> dict[str, Any]:
        """Return the execution trace in response-ready form."""
        return {
            "agent": "BreastCareAgent",
            "steps": [step.to_dict() for step in self.trace],
        }
