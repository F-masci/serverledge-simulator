"""Completamento di una richiesta accettata."""

from dataclasses import dataclass

from ..request import Request, RequestStatus
from .event import Event

@dataclass(frozen=True)
class CompletionEvent(Event):
    request: Request

    def execute(self) -> None:
        """Conclude la richiesta al tempo corrente del motore."""
        now = self.ctx.engine.now
        request = self.request
        assert request.node is not None and request.container is not None
        if request.status is not RequestStatus.ACCEPTED:
            raise ValueError('Solo una richiesta accettata può essere completata')

        # Registra la conclusione della richiesta nei metriche e notifica l'osservatore.
        self._record_completion(request)
        self.request.complete(now)

    def _record_completion(self, request: Request) -> None:
        """Registra la conclusione della richiesta nei metriche e notifica l'osservatore."""
        now = self.ctx.engine.now
        self.ctx.metrics.record_completion(request, now)
        self.ctx.observer.notify_request_completed(request)