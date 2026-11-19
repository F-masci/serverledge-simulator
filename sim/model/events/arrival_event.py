"""Arrivo di una richiesta di invocazione."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..function import Function
from ..request import Request
from .completion_event import CompletionEvent
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
        duration = sampler.service.sample()
        request = Request(self.function, service_duration=duration)
        node = infrastructure.submit(request)
        if node is not None:
            if request.warm is False:
                # Il cold start è istantaneo finché lo scenario non ne definisce la durata.
                assert request.container is not None
                node.creation_succeeded(request.container, engine.now)
            engine.schedule_after(duration, CompletionEvent(request, arrival_time=engine.now))
        simulation.metrics.record_admission(request, engine.now)
