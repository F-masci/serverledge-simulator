"""Completamento di una richiesta accettata."""

from dataclasses import dataclass

from ..request import Request
from .event import Event

@dataclass(frozen=True)
class CompletionEvent(Event):
    request: Request
    arrival_time: float

    def execute(self) -> None:
        """Conclude la richiesta al tempo corrente del motore."""
        self.request.complete(self.ctx.engine.now)
        self.ctx.simulation.record_completion(self.request, self.arrival_time)
        assert self.request.node is not None
        self.ctx.simulation.dispatch_waiting(self.request.node)
