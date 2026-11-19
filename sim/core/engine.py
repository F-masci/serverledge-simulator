"""Motore sequenziale a eventi discreti con calendario deterministico."""

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import count
from typing import TYPE_CHECKING

from sim.utils.validation import finite_time, finite_delay

from sim.model.events import Event, EventStatus, ScheduledEvent


if TYPE_CHECKING:
    from .simulation import SimulationContext


@dataclass(order=True, frozen=True, slots=True)
class _CalendarEntry:
    """Voce ordinata per tempo simulato e sequenza di inserimento."""

    time: float
    sequence: int
    scheduled_event: ScheduledEvent = field(compare=False)


class SimulationEngine:
    """Orologio in secondi simulati, calendario e dispatch degli eventi.

    A parità di tempo gli eventi seguono l'ordine di inserimento. Il motore
    non confronta oggetti Event e non contiene logica di servizio o routing.
    """

    def __init__(self, ctx: 'SimulationContext') -> None:
        # Contesto della simulazione, che fornisce accesso all'infrastruttura e alle metriche
        self.ctx = ctx

        # Tempo simulato in secondi, inizializzato a zero
        self._now: float = 0.0

        # Calendario degli eventi, implementato come una coda di priorità (heap)
        self._calendar: list[_CalendarEntry] = []

        # Sequenza per distinguere eventi con lo stesso tempo
        self._sequence = count()

        # Flag per evitare l'esecuzione ricorsiva del motore
        self._executing = False

    @property
    def now(self) -> float:
        return self._now

    def schedule(self, time: float, event: Event) -> ScheduledEvent:
        """Pianifica a un tempo assoluto, anche uguale a now, mai nel passato."""
        time = finite_time(time, 'time')
        if time < self.now:
            raise ValueError('Impossibile pianificare nel passato')
        if not isinstance(event, Event):
            raise TypeError('event deve implementare Event')

        # La sequenza viene generata per garantire
        # l'ordine di inserimento degli eventi con
        # lo stesso tempo.
        seq_number = next(self._sequence)

        scheduled_event = ScheduledEvent(time, event, ctx=self.ctx)
        calendar_entry = _CalendarEntry(time, seq_number, scheduled_event)
        heappush(self._calendar, calendar_entry)

        return scheduled_event

    def schedule_after(self, delay: float, event: Event) -> ScheduledEvent:
        """Pianifica relativamente al tempo corrente."""
        delay = finite_delay(delay, 'delay')
        return self.schedule(self.now + delay, event)

    def _discard_cancelled(self) -> None:
        """Rimuove le occorrenze cancellate dal calendario."""
        if not self._calendar:
            return

        # Cancellazione lazy: le occorrenze vengono rimosse quando affiorano.
        while self._calendar and self._calendar[0].scheduled_event.status is EventStatus.CANCELLED:
            heappop(self._calendar)

    def _execute_next(self) -> bool:
        """Esegue l'evento successivo; False se il calendario è esaurito."""
        self._discard_cancelled()
        if not self._calendar:
            return False

        # Recupera l'evento successivo dal calendario e aggiorna il tempo simulato
        entry = heappop(self._calendar)
        self._now = entry.time

        entry.scheduled_event.execute()
        return True

    def step(self) -> bool:
        """Esegue un evento attivo; False se il calendario è esaurito."""
        if self._executing:
            raise RuntimeError('Esecuzione ricorsiva del motore non consentita')
        self._executing = True
        try:
            return self._execute_next()
        finally:
            self._executing = False
