"""Motore sequenziale a eventi discreti con calendario deterministico."""

from heapq import heappop, heappush
from itertools import count
from typing import TYPE_CHECKING

from sim.utils.validation import finite_time

from sim.model.events import Event, EventStatus, ScheduledEvent


if TYPE_CHECKING:
    from sim.model.infrastructure import Infrastructure
    from .simulation import Simulation


class SimulationEngine:
    """Orologio in secondi simulati, calendario e dispatch degli eventi.

    A parità di tempo gli eventi seguono l'ordine di inserimento. Il motore
    non confronta oggetti Event e non contiene logica di servizio o routing.
    """

    def __init__(
        self, *, simulation: 'Simulation | None' = None,
        infrastructure: 'Infrastructure | None' = None,
    ) -> None:
        self._simulation = simulation
        self._infrastructure = infrastructure
        self._now = 0.0
        self._calendar: list[tuple[float, int, ScheduledEvent]] = []
        self._sequence = count()
        self._executing = False

    @property
    def now(self) -> float:
        return self._now

    @property
    def pending_count(self) -> int:
        """Numero delle occorrenze attive, escluse quelle cancellate."""
        return sum(handle.status is EventStatus.PENDING for _, _, handle in self._calendar)

    @property
    def next_event_time(self) -> float | None:
        self._discard_cancelled()
        return self._calendar[0][0] if self._calendar else None

    def schedule(self, time: float, event: Event) -> ScheduledEvent:
        """Pianifica a un tempo assoluto, anche uguale a now, mai nel passato."""
        time = finite_time(time, 'time')
        if time < self.now:
            raise ValueError('Impossibile pianificare nel passato')
        if not isinstance(event, Event):
            raise TypeError('event deve implementare Event')
        handle = ScheduledEvent(time, event)
        heappush(self._calendar, (time, next(self._sequence), handle))
        return handle

    def schedule_after(self, delay: float, event: Event) -> ScheduledEvent:
        """Pianifica relativamente al tempo corrente."""
        delay = finite_time(delay, 'delay')
        if delay < 0:
            raise ValueError('Il ritardo non può essere negativo')
        return self.schedule(self.now + delay, event)

    def _discard_cancelled(self) -> None:
        # Cancellazione lazy: le occorrenze vengono rimosse quando affiorano.
        while self._calendar and self._calendar[0][2].status is EventStatus.CANCELLED:
            heappop(self._calendar)

    def _execute_next(self) -> bool:
        self._discard_cancelled()
        if not self._calendar:
            return False
        time, _, handle = heappop(self._calendar)
        self._now = time
        handle._execute(self._simulation, self, self._infrastructure)
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

    def run(self, until: float | None = None) -> int:
        """Esegue fino a esaurimento o fino a until incluso; restituisce il conteggio.

        Con until esplicito l'orologio raggiunge l'orizzonte anche senza eventi.
        Gli eventi successivi restano disponibili per una chiamata successiva.
        Le eccezioni interrompono run e vengono propagate senza rollback.
        """
        if self._executing:
            raise RuntimeError('Esecuzione ricorsiva del motore non consentita')
        if until is not None:
            until = finite_time(until, 'until')
            if until < self.now:
                raise ValueError('L\'orizzonte non può precedere il tempo corrente')
        self._executing = True
        executed = 0
        try:
            while (next_time := self.next_event_time) is not None:
                if until is not None and next_time > until:
                    break
                self._execute_next()
                executed += 1
            if until is not None:
                self._now = until
            return executed
        finally:
            self._executing = False
