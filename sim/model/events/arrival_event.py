"""Arrivo di una richiesta di invocazione."""

from dataclasses import dataclass
from typing import Callable

from ..function import Function
from ..node import Node
from ..region import Region
from ..request import Request, RequestStatus
from .event import Event


@dataclass(frozen=True)
class ArrivalEvent(Event):
    function: Function
    entry_region: Region
    entry_node: 'Node | None'

    ### Callback ###
    on_request_submitted: Callable[[Request], None] | None = None
    on_request_accepted: Callable[[Request], None] | None = None
    on_request_rejected: Callable[[Request], None] | None = None
    on_request_queued: Callable[[Request], None] | None = None

    #################################
    ### Internal state management ###
    #################################

    ## Arrival management ##

    def execute(self) -> None:
        """Pianifica il prossimo arrivo e invia la richiesta alla regione di ingresso."""
        engine = self.ctx.engine
        sampler = self.function.sampler
        if sampler is None:
            raise RuntimeError('La funzione dell’arrivo deve avere un campionatore')

        # TODO: campionare servizio e cold start dopo il placement, da profili
        # calibrati per funzione e destinazione (architettura, capacità e carico).
        # Calibrare anche la latenza di offloading per tratta con misure del reale.
        arrival_time = engine.now
        service_duration = sampler.service.sample()
        can_offload = self.ctx.scenario.invocation.can_offload

        # Crea la richiesta con i parametri campionati.
        request = Request(
            self.function, service_duration=service_duration, arrival_time=arrival_time,
            can_offload=can_offload,
        )

        # Notifica l'arrivo della richiesta per la funzione nella regione di ingresso.
        self.ctx.observer.notify_arrival(request, region=self.entry_region)


        # Pianifica il prossimo arrivo secondo l'interarrivo campionato.
        next_time = engine.now + sampler.arrivals.next_iat()
        if next_time <= engine.now:
            raise ValueError('Interarrivo troppo piccolo per avanzare il tempo simulato')
        engine.schedule(next_time, self)


        # Inoltra la richiesta nella regione di ingresso.
        self.entry_region.submit(request, now=engine.now, node=self.entry_node)
        self._handle_callback(request)

    ## Callback handling ##

    def _handle_callback(self, request: Request) -> None:
        """Invoca il callback corrispondente allo stato della richiesta."""
        if self.on_request_submitted is not None:
            self.on_request_submitted(request)

        if request.status is RequestStatus.ACCEPTED:
            if self.on_request_accepted is not None:
                self.on_request_accepted(request)
        elif request.status is RequestStatus.REJECTED:
            if self.on_request_rejected is not None:
                self.on_request_rejected(request)
        elif request.status is RequestStatus.QUEUED:
            if self.on_request_queued is not None:
                self.on_request_queued(request)