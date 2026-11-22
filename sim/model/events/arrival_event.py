"""Arrivo di una richiesta di invocazione."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .event import Event
from ..function import Function
from ..node import Node
from ..region import Region
from ..request import Request

if TYPE_CHECKING:
    from sim.bindings import RequestBindings


@dataclass(frozen=True)
class ArrivalEvent(Event):
    function: Function
    entry_region: Region
    entry_node: 'Node | None'
    request_bindings: 'RequestBindings | None' = None

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

        # Crea la richiesta con i parametri campionati e applica subito i suoi binding.
        request = Request(
            self.function, service_duration=service_duration, arrival_time=arrival_time,
            can_offload=can_offload,
        )
        if self.request_bindings is not None:
            self.request_bindings.bind(request)

        # Notifica l'arrivo della richiesta per la funzione nella regione di ingresso.
        self.ctx.observer.notify_arrival(request, region=self.entry_region)

        # Pianifica il prossimo arrivo secondo l'interarrivo campionato.
        next_time = engine.now + sampler.arrivals.next_iat()
        if next_time <= engine.now:
            raise ValueError('Interarrivo troppo piccolo per avanzare il tempo simulato')
        engine.schedule(next_time, self)

        # Inoltra la richiesta nella regione di ingresso.
        self.entry_region.submit(request, now=engine.now, node=self.entry_node)