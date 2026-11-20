"""Completamento di una richiesta accettata."""

from dataclasses import dataclass

from ..request import Request, RequestStatus
from .event import Event


@dataclass(frozen=True)
class CompletionEvent(Event):
    request: Request

    def execute(self) -> None:
        """Termina il servizio e pianifica l'eventuale ritorno della risposta."""
        now = self.ctx.engine.now
        request = self.request
        if request.status is not RequestStatus.ACCEPTED:
            raise ValueError('Solo una richiesta accettata può essere completata')
        if request.return_delay > 0:
            request.begin_response(now)
            self.ctx.engine.schedule_after(request.return_delay, ResponseEvent(request, ctx=self.ctx))
            return

        self.ctx.metrics.record_completion(request, now)
        self.ctx.observer.notify_request_completed(request)
        request.complete(now)


@dataclass(frozen=True)
class ResponseEvent(Event):
    request: Request

    def execute(self) -> None:
        if self.request.status is not RequestStatus.RESPONDING:
            raise ValueError('La richiesta non ha una risposta in transito')
        self.ctx.metrics.record_completion(self.request, self.ctx.engine.now)
        self.ctx.observer.notify_request_completed(self.request)
        self.request.finish_response()
