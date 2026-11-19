"""Contratto comune degli eventi di simulazione."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.core.simulation import Simulation
    from ..infrastructure import Infrastructure


class Event(ABC):
    """Operazione istantanea eseguita al tempo corrente del motore.

    Le durate si rappresentano pianificando eventi futuri, senza attese reali.
    """

    @abstractmethod
    def execute(
        self, simulation: 'Simulation | None', engine: 'SimulationEngine',
        infrastructure: 'Infrastructure | None',
    ) -> None:
        """Modifica il modello e può pianificare ulteriori eventi."""
