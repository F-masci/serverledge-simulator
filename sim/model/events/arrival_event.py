"""Arrivo di una richiesta di invocazione."""

from dataclasses import dataclass

from ..function import Function
from ..request import Request
from .event import Event


@dataclass(frozen=True)
class ArrivalEvent(Event):
    function: Function

    def execute(self) -> None:
        """Pianifica il prossimo arrivo e invia la richiesta alla regione di ingresso."""
        engine = self.ctx.engine
        sampler = self.function.sampler
        if sampler is None:
            raise RuntimeError('La funzione dell’arrivo deve avere un campionatore')

        next_time = engine.now + sampler.arrivals.next_iat()
        if next_time <= engine.now:
            raise ValueError('Interarrivo troppo piccolo per avanzare il tempo simulato')
        engine.schedule(next_time, self)

        # TODO: campionare servizio e cold start dopo il placement, da profili
        # calibrati per funzione e destinazione (architettura, capacità e carico).
        # Calibrare anche la latenza di offloading per tratta con misure del reale.
        request = Request(
            self.function, service_duration=sampler.service.sample(), arrival_time=engine.now,
            can_offload=self.ctx.scenario.invocation.can_offload,
        )
        self.ctx.simulation.submit(request)
