"""Completamento di una richiesta accettata."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..request import Request
from .event import Event

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.core.simulation import Simulation
    from ..infrastructure import Infrastructure


@dataclass(frozen=True)
class CompletionEvent(Event):
    request: Request

    def execute(
        self, simulation: 'Simulation | None', engine: 'SimulationEngine',
        infrastructure: 'Infrastructure | None',
    ) -> None:
        """Conclude la richiesta al tempo corrente del motore."""
        self.request.complete(engine.now)
