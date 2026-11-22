"""Completamento di una richiesta accettata."""

from dataclasses import dataclass

from .event import Event
from ..request import Request


@dataclass(frozen=True)
class CompletionEvent(Event):
    """Evento di fine servizio di una richiesta accettata."""

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
    """Evento di consegna al client della risposta (o del rifiuto) da una regione remota."""

    request: Request

    def execute(self) -> None:
        """Completa la risposta o il rifiuto della richiesta."""
        request = self.request
        if not request.is_responding:
            raise ValueError('La richiesta non ha una risposta in transito')

        # Distingue un rifiuto remoto in transito da un completamento ordinario.
        if request.rejection_reason is not None:
            request.finish_rejection()
            return

        # Completa la risposta della richiesta; on_completed notifica l'osservatore.
        request.finish_response()
