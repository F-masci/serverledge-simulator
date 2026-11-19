"""Fine della creazione di un container riservato a una richiesta."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..request import Request
from .completion_event import CompletionEvent
from .event import Event

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.core.simulation import Simulation
    from ..infrastructure import Infrastructure


@dataclass(frozen=True)
class ContainerReadyEvent(Event):
    request: Request
    arrival_time: float

    def execute(
        self, simulation: 'Simulation | None', engine: 'SimulationEngine',
        infrastructure: 'Infrastructure | None',
    ) -> None:
        node, container = self.request.node, self.request.container
        if node is None or container is None:
            raise ValueError('La richiesta non ha un container assegnato')
        duration = self.request.service_duration
        if duration is None:
            raise ValueError('La richiesta deve avere una durata di servizio')
        node.creation_succeeded(container, engine.now)
        engine.schedule_after(duration, CompletionEvent(self.request, self.arrival_time))
