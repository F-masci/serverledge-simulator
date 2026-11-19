"""Controllo del janitor su un container idle."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .event import Event

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.core.simulation import Simulation
    from ..container import DockerContainer
    from ..infrastructure import Infrastructure
    from ..node import Node


@dataclass(frozen=True)
class ContainerCleanupEvent(Event):
    node: 'Node'
    container: 'DockerContainer'

    def execute(
        self, simulation: 'Simulation | None', engine: 'SimulationEngine',
        infrastructure: 'Infrastructure | None',
    ) -> None:
        self.node.expire(engine.now, container=self.container)
