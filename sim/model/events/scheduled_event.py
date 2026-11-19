"""Occorrenze degli eventi nel calendario e relativi stati."""

from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from .event import Event

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext


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

    def __init__(
        self, time: float, event: Event, *, ctx: 'SimulationContext',
        on_status_change: 'Callable[[ScheduledEvent, EventStatus], None] | None' = None,
    ) -> None:
        if event.ctx is not ctx:
            raise ValueError('Evento e calendario devono condividere lo stesso contesto')
        self._time = time
        self._event = event
        self._status = EventStatus.PENDING
        self._on_status_change = on_status_change

    @property
    def time(self) -> float:
        return self._time

    @property
    def event(self) -> Event:
        return self._event

    @property
    def status(self) -> EventStatus:
        return self._status

    def _set_status(self, status: EventStatus) -> None:
        previous = self._status
        self._status = status
        if self._on_status_change is not None:
            self._on_status_change(self, previous)

    def cancel(self) -> bool:
        """True solo se l'occorrenza era ancora pendente."""
        if self.status is not EventStatus.PENDING:
            return False
        self._set_status(EventStatus.CANCELLED)
        return True

    def execute(self) -> None:
        self._set_status(EventStatus.RUNNING)
        try:
            self.event.execute()
        except BaseException:
            self._set_status(EventStatus.FAILED)
            raise
        self._set_status(EventStatus.EXECUTED)
