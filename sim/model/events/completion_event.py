"""Completamento di una richiesta accettata."""

from dataclasses import dataclass

from .event import Event
from ..request import Request


@dataclass(frozen=True)
class CompletionEvent(Event):
    request: Request

    def execute(self) -> None:
        """Termina il servizio e pianifica l'eventuale ritorno della risposta."""
        now = self.ctx.engine.now
        request = self.request
        if not request.is_accepted:
            raise ValueError('Solo una richiesta accettata può essere completata')

        # Se la richiesta ha un ritardo di ritorno,
        # pianifica l'evento di risposta dopo il ritardo.
        if request.return_delay > 0:
            request.begin_response(now)
            self.ctx.engine.schedule_after(request.return_delay, ResponseEvent(ctx=self.ctx, request=request))
            return

        # Se la richiesta non ha un ritardo di ritorno, completa immediatamente la richiesta.
        # La notifica di completamento viene invocata dalla callback on_completed della richiesta.
        request.complete(now)


@dataclass(frozen=True)
class ResponseEvent(Event):
    request: Request

    def execute(self) -> None:
        """Completa la risposta della richiesta."""
        request = self.request
        if not request.is_responding:
            raise ValueError('La richiesta non ha una risposta in transito')

        # Completa la risposta della richiesta; on_completed notifica l'osservatore.
        request.finish_response()
