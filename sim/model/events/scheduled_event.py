"""Occorrenze degli eventi nel calendario e relativi stati."""

from enum import Enum
from typing import TYPE_CHECKING

from .event import Event

if TYPE_CHECKING:
    from sim.core.engine import SimulationEngine
    from sim.core.simulation import Simulation
    from ..infrastructure import Infrastructure


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

    def _execute(
        self, simulation: 'Simulation | None', engine: 'SimulationEngine',
        infrastructure: 'Infrastructure | None',
    ) -> None:
        self._status = EventStatus.RUNNING
        try:
            self.event.execute(simulation, engine, infrastructure)
        except BaseException:
            self._status = EventStatus.FAILED
            raise
        self._status = EventStatus.EXECUTED
