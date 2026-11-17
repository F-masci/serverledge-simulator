"""Contratto polimorfico degli eventi e stato delle loro pianificazioni."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import SimulationEngine


class Event(ABC):
    """Operazione istantanea eseguita al tempo corrente del motore.

    Le durate si rappresentano pianificando eventi futuri, senza attese reali.
    """

    @abstractmethod
    def execute(self, engine: 'SimulationEngine') -> None:
        """Modifica il modello e può pianificare ulteriori eventi."""


class EventStatus(Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    EXECUTED = 'executed'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


class ScheduledEvent:
    """Singola occorrenza nel calendario, distinta dall'oggetto Event.

    La cancellazione agisce su questa occorrenza; non su altre pianificazioni
    dello stesso oggetto. Il tempo e lo stato sono pubblicamente di sola lettura.
    """

    def __init__(self, time: float, event: Event) -> None:
        self._time = time
        self._event = event
        self._status = EventStatus.PENDING

    @property
    def time(self) -> float:
        return self._time

    @property
    def event(self) -> Event:
        return self._event

    @property
    def status(self) -> EventStatus:
        return self._status

    def cancel(self) -> bool:
        """True solo se l'occorrenza era ancora pendente."""
        if self.status is not EventStatus.PENDING:
            return False
        self._status = EventStatus.CANCELLED
        return True

    def _execute(self, engine: 'SimulationEngine') -> None:
        self._status = EventStatus.RUNNING
        try:
            self.event.execute(engine)
        except BaseException:
            self._status = EventStatus.FAILED
            raise
        self._status = EventStatus.EXECUTED
