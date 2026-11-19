"""Contratto comune degli eventi di simulazione."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext


@dataclass(frozen=True, kw_only=True)
class Event(ABC):
    """Operazione istantanea eseguita al tempo corrente del motore.

    Le durate si rappresentano pianificando eventi futuri, senza attese reali.
    """

    ctx: 'SimulationContext' = field(repr=False, compare=False)

    @abstractmethod
    def execute(self) -> None:
        """Modifica il modello e può pianificare ulteriori eventi."""
