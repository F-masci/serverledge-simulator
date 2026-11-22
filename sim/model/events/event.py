"""Contratto comune degli eventi di simulazione."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext


@dataclass(frozen=True, kw_only=True)
class Event(ABC):
    """
    Contratto comune degli eventi di simulazione.

    Un evento di simulazione rappresenta un'azione che modifica lo stato del modello
    e può pianificare ulteriori eventi.
    """

    ctx: 'SimulationContext' = field(repr=False, compare=False)

    @abstractmethod
    def execute(self) -> None:
        """Modifica il modello e può pianificare ulteriori eventi."""