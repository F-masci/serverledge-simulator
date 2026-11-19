"""Arrivo di una richiesta di invocazione."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..function import Function
from ..request import Request
from .event import Event

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.core.simulation import Simulation
    from ..infrastructure import Infrastructure


@dataclass(frozen=True)
class ArrivalEvent(Event):
    function: Function

    def execute(
        self, simulation: 'Simulation | None', engine: 'SimulationEngine',
        infrastructure: 'Infrastructure | None',
    ) -> None:
        """Schedula il prossimo arrivo e consegna la richiesta corrente."""
        if simulation is None or infrastructure is None:
            raise RuntimeError('ArrivalEvent richiede una simulazione e un’infrastruttura')
        sampler = self.function.sampler
        if sampler is None:
            raise RuntimeError('La funzione dell’arrivo deve avere un campionatore')
        next_time = engine.now + sampler.arrivals.next_iat()
        if next_time <= engine.now:
            raise ValueError('Interarrivo troppo piccolo per avanzare il tempo simulato')
        engine.schedule(next_time, ArrivalEvent(self.function))
        infrastructure.submit(Request(self.function, service_duration=sampler.service.sample()))
        simulation.record_arrival()
