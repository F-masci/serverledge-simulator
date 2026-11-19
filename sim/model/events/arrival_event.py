"""Arrivo di una richiesta di invocazione."""

from dataclasses import dataclass
from collections.abc import Callable

from ..function import Function
from ..request import Request, RequestStatus
from ..node import Node
from .event import Event

@dataclass(frozen=True)
class ArrivalEvent(Event):
    function: Function

    ### Callback ###
    on_accepted: Callable[[Request], None] | None = None
    on_rejected: Callable[[Request], None] | None = None

    def execute(self) -> None:
        """Schedula il prossimo arrivo e consegna la richiesta corrente."""
        engine = self.ctx.engine
        sampler = self.function.sampler
        if sampler is None:
            raise RuntimeError('La funzione dell’arrivo deve avere un campionatore')

        # Schedula il prossimo arrivo prima di consegnare la richiesta corrente.
        self._schedule_next_arrival()

        # Consegna la richiesta corrente al sistema.
        request = self._submit_request()

        # Gestisce l’esito della consegna della richiesta.
        if request.status is RequestStatus.ACCEPTED:
            # La richiesta è stata accettata e assegnata a un nodo.
            self._handle_accepted(request)
        else:
            # La richiesta è stata rifiutata.
            self._handle_rejected(request)

    def _schedule_next_arrival(self) -> None:
        """Schedula il prossimo arrivo della funzione associata all’evento."""
        engine = self.ctx.engine
        sampler = self.function.sampler
        if sampler is None:
            raise RuntimeError('La funzione dell’arrivo deve avere un campionatore')

        # Calcola il tempo del prossimo arrivo.
        next_time = engine.now + sampler.arrivals.next_iat()
        if next_time <= engine.now:
            raise ValueError('Interarrivo troppo piccolo per avanzare il tempo simulato')

        # Schedula il prossimo evento di arrivo.
        event = ArrivalEvent(self.function, on_accepted=self.on_accepted, ctx=self.ctx)
        engine.schedule(next_time, event)

    def _submit_request(self) -> Request:
        """Invia la richiesta al sistema e restituisce il nodo assegnato o None se rifiutata."""
        engine = self.ctx.engine
        sampler = self.function.sampler
        if sampler is None:
            raise RuntimeError('La funzione dell’arrivo deve avere un campionatore')

        duration = sampler.service.sample()
        request = Request(self.function, service_duration=duration, arrival_time=engine.now)
        self.ctx.infrastructure.submit(request)
        return request

    def _handle_accepted(self, request: Request) -> None:
        """Gestisce l’accettazione della richiesta da parte del nodo."""
        now = self.ctx.engine.now
        self.ctx.metrics.record_submission(request, now)
        self.ctx.observer.notify_request_accepted(request)

        # Schedula l’esecuzione della richiesta sul nodo assegnato
        # tramite la callback on_accepted su Simulation.
        if self.on_accepted is not None:
            self.on_accepted(request)

    def _handle_rejected(self, request: Request) -> None:
        """Gestisce il rifiuto della richiesta da parte del nodo."""
        now = self.ctx.engine.now
        self.ctx.metrics.record_submission(request, now)
        self.ctx.observer.notify_request_rejected(request)
        if self.on_rejected is not None:
            self.on_rejected(request)
