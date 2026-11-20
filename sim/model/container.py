"""Stato logico di un container Docker, senza accesso al daemon."""
from collections.abc import Callable
from enum import Enum
import json
from .function import Function
from ..utils.validation import finite_time


class ContainerState(Enum):
    """Stato logico di un container Docker."""
    CREATING = 'creating'
    BUSY = 'busy'
    IDLE = 'idle'
    DESTROYED = 'destroyed'


class DockerContainer:
    """
    Rappresenta lo stato di un container Docker
    """

    def __init__(self, container_id: str, function: Function) -> None:
        self._id = container_id
        self._function = function
        self._state = ContainerState.CREATING
        self._requests_count = 0
        self._expiration_time: float | None = None

        ### Callback ###
        
        self.on_created: Callable[[DockerContainer], None] | None = None
        self.on_busy: Callable[[DockerContainer], None] | None = None
        self.on_idle: Callable[[DockerContainer], None] | None = None
        self.on_destroyed: Callable[[DockerContainer], None] | None = None
        
        self.on_state_changed: Callable[[DockerContainer, ContainerState, ContainerState], None] | None = None

    @property
    def id(self) -> str:
        return self._id

    @property
    def function(self) -> Function:
        return self._function

    @property
    def state(self) -> ContainerState:
        return self._state

    @property
    def requests_count(self) -> int:
        return self._requests_count

    @property
    def expiration_time(self) -> float | None:
        """Tempo simulato in secondi"""
        return self._expiration_time

    @property
    def memory_limit_bytes(self) -> int:
        return self.function.memory_mb * 1048576

    @property
    def cpu_period_us(self) -> int:
        return 50000

    @property
    def cpu_quota_us(self) -> int:
        return int(self.cpu_period_us * self.function.cpu_demand)
    
    def is_expired(self, now: float) -> bool:
        """Verifica se il container è scaduto."""
        now = finite_time(now, 'now')
        expiration_time = self.expiration_time
    
        # Un container è considerato scaduto se è in stato IDLE
        # e il tempo simulato corrente supera il tempo di scadenza.
        if self._state is not ContainerState.IDLE:
            return False
        if expiration_time is None:
            return False
        return now > expiration_time

    def created(self, *, idle: bool, now: float, timeout: float) -> None:
        """Segna il container come creato, pronto per l'uso."""
        if self._state is not ContainerState.CREATING:
            raise ValueError('Container non in creazione')
        
        # Imposta il contatore e la scadenza prima di notificare la creazione.
        state = ContainerState.IDLE if idle else ContainerState.BUSY
        self._requests_count = 0 if idle else 1
        self._expiration_time = now + timeout if idle else None
        self._handle_change_status(state, self.on_created)

        # Dopo la creazione, notifica anche lo stato operativo iniziale.
        if self.on_idle is not None and self._state is ContainerState.IDLE:
            self.on_idle(self)
        if self.on_busy is not None and self._state is ContainerState.BUSY:
            self.on_busy(self)

    def acquire(self) -> None:
        """Segna il container come occupato da una richiesta."""
        if self._state not in (ContainerState.IDLE, ContainerState.BUSY):
            raise ValueError('Container non riutilizzabile')
        if self.requests_count >= self.function.max_concurrency:
            raise ValueError('Concorrenza massima raggiunta')
        
        # Ogni acquisizione notifica on_busy, anche con altre richieste attive.
        self._requests_count += 1
        self._expiration_time = None
        self._handle_change_status(ContainerState.BUSY, self.on_busy)

    def complete(self, now: float, timeout: float) -> None:
        """Segna il completamento di una richiesta e aggiorna lo stato del container."""
        if self._state is not ContainerState.BUSY:
            raise ValueError('Nessuna richiesta attiva')

        # Il container torna idle alla conclusione dell'ultima richiesta attiva.
        self._requests_count -= 1
        if self.requests_count == 0:
            self._expiration_time = now + timeout
            self._handle_change_status(ContainerState.IDLE, self.on_idle)

    def destroy(self) -> None:
        """Segna il container come distrutto."""
        if self._state not in (ContainerState.IDLE, ContainerState.CREATING):
            raise ValueError('Distruzione ordinaria consentita solo per idle o creazione fallita')

        self._handle_change_status(ContainerState.DESTROYED, self.on_destroyed)

    def _handle_change_status(
        self, state: ContainerState, callback: Callable[[DockerContainer], None] | None,
    ) -> None:
        """Aggiorna lo stato e invoca le callback nell'ordine previsto."""
        previous_state = self._state
        self._state = state

        # La callback specifica precede la notifica del cambio di stato.
        if callback is not None:
            callback(self)
        if self.on_state_changed is not None and previous_state is not state:
            self.on_state_changed(self, previous_state, state)

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile del container."""
        fields = self.to_dict()
        details = ', '.join(f'{name}={value}' for name, value in fields.items() if value is not None)
        return f'DockerContainer({details})'

    def to_dict(self) -> dict:
        """Restituisce lo stato corrente del container in un formato serializzabile."""
        return {
            'id': self.id,
            'function': self.function.name,
            'state': self.state.value,
            'requests_count': self.requests_count,
            'expiration_time_s': self.expiration_time,
        }

    def to_json(self) -> str:
        """Serializza il container per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'DockerContainer({self.id!r}, {self._state.value}, requests={self.requests_count})'
