"""Occorrenze degli eventi nel calendario e relativi stati."""

from enum import Enum
from typing import TYPE_CHECKING

from .event import Event

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext


class EventStatus(Enum):
    """Stati di un'occorrenza di evento nel calendario."""
    PENDING = 'pending'    # In attesa di essere eseguita dal motore.
    RUNNING = 'running'    # In corso di esecuzione.
    EXECUTED = 'executed'  # Eseguita con successo.
    FAILED = 'failed'      # Terminata con un'eccezione durante l'esecuzione.


class ScheduledEvent:
    """
    Singola occorrenza di un evento nel calendario di simulazione.

    Avvolge un oggetto Event tracciandone lo stato di esecuzione
    all'interno del contesto della replica.
    """

    def __init__(self, event: Event, *, ctx: 'SimulationContext') -> None:
        # Verifica che l'evento appartenga allo stesso contesto del calendario.
        if event.ctx is not ctx:
            raise ValueError('Evento e calendario devono condividere lo stesso contesto')

        # Evento associato e stato iniziale dell'occorrenza.
        self._event = event
        self._status = EventStatus.PENDING

    ### Event State ###

    @property
    def event(self) -> Event:
        return self._event

    @property
    def status(self) -> EventStatus:
        return self._status

    @property
    def is_pending(self) -> bool:
        """Indica se l'occorrenza è in attesa di essere eseguita."""
        return self._status is EventStatus.PENDING

    @property
    def is_running(self) -> bool:
        """Indica se l'occorrenza è attualmente in esecuzione."""
        return self._status is EventStatus.RUNNING

    @property
    def is_executed(self) -> bool:
        """Indica se l'occorrenza è stata eseguita con successo."""
        return self._status is EventStatus.EXECUTED

    @property
    def is_failed(self) -> bool:
        """Indica se l'esecuzione dell'occorrenza è terminata con un'eccezione."""
        return self._status is EventStatus.FAILED

    #################################
    ### Internal state management ###
    #################################

    ## Event execution ##

    def execute(self) -> None:
        """Esegue l'evento associato aggiornando lo stato dell'occorrenza."""
        # Segna l'occorrenza come in esecuzione prima di invocare l'evento.
        self._status = EventStatus.RUNNING
        try:
            self.event.execute()
        except BaseException:
            # In caso di errore registra il fallimento e propaga l'eccezione.
            self._status = EventStatus.FAILED
            raise

        # Marca l'occorrenza come completata al termine dell'esecuzione.
        self._status = EventStatus.EXECUTED
