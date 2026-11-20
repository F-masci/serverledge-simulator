"""Stato logico di un container Docker, senza accesso al daemon."""
from collections.abc import Callable
from enum import Enum
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
        
        # Aggiorna lo stato e il contatore delle richieste in base al parametro idle
        self._state = ContainerState.IDLE if idle else ContainerState.BUSY
        self._requests_count = 0 if idle else 1
        self._expiration_time = now + timeout if idle else None
        
        # Invoca i callback se definiti
        if self.on_created:
            self.on_created(self)
        if self.on_state_changed:
            self.on_state_changed(self, ContainerState.CREATING, self._state)
        if self.on_idle and self._state is ContainerState.IDLE:
            self.on_idle(self)
        if self.on_busy and self._state is ContainerState.BUSY:
            self.on_busy(self)
        

    def acquire(self) -> None:
        """Segna il container come occupato da una richiesta."""
        if self._state not in (ContainerState.IDLE, ContainerState.BUSY):
            raise ValueError('Container non riutilizzabile')
        if self.requests_count >= self.function.max_concurrency:
            raise ValueError('Concorrenza massima raggiunta')
        
        # Aggiorna lo stato e il contatore delle richieste
        self._requests_count += 1
        self._state = ContainerState.BUSY
        self._expiration_time = None
        
        # Invoca i callback se definiti
        if self.on_busy:
            self.on_busy(self)
        if self.on_state_changed and self._requests_count == 1:  # Solo se il container era idle prima
            self.on_state_changed(self, ContainerState.IDLE, self._state)

    def complete(self, now: float, timeout: float) -> None:
        """Segna il completamento di una richiesta e aggiorna lo stato del container."""
        if self._state is not ContainerState.BUSY:
            raise ValueError('Nessuna richiesta attiva')
        
        # Aggiorna il contatore delle richieste e lo stato del container
        self._requests_count -= 1
        if self.requests_count == 0:
            self._state = ContainerState.IDLE
            self._expiration_time = now + timeout
            
        # Invoca i callback se definiti
        if self.on_idle and self._state is ContainerState.IDLE:
            self.on_idle(self)
        if self.on_state_changed and self.requests_count == 0:  # Solo se il container era busy prima
            self.on_state_changed(self, ContainerState.BUSY, self._state)

    def destroy(self) -> None:
        """Segna il container come distrutto."""
        if self._state not in (ContainerState.IDLE, ContainerState.CREATING):
            raise ValueError('Distruzione ordinaria consentita solo per idle o creazione fallita')
        
        # Aggiorna lo stato del container a DESTROYED
        prevState = self._state
        self._state = ContainerState.DESTROYED
        
        # Invoca i callback se definiti
        if self.on_destroyed:
            self.on_destroyed(self)
        if self.on_state_changed:
            self.on_state_changed(self, prevState, self._state)

    def __repr__(self) -> str:
        return f'DockerContainer({self.id!r}, {self._state.value}, requests={self.requests_count})'
