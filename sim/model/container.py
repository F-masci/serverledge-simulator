"""Stato logico di un container Docker, senza accesso al daemon."""

from enum import Enum
from .function import Function


class ContainerState(Enum):
    CREATING = 'creating'
    BUSY = 'busy'
    IDLE = 'idle'
    DESTROYED = 'destroyed'


class DockerContainer:
    """Identità per istanza, come i puntatori dei pool Go.

    CREATING e DESTROYED sono stati espliciti del modello simulato.
    """

    def __init__(self, container_id: str, function: Function) -> None:
        self._id = container_id
        self._function = function
        self._state = ContainerState.CREATING
        self._requests_count = 0
        self._expiration_time: float | None = None

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

    def _created(self, *, idle: bool, now: float, timeout: float) -> None:
        if self.state is not ContainerState.CREATING:
            raise ValueError('Container non in creazione')
        self._state = ContainerState.IDLE if idle else ContainerState.BUSY
        self._requests_count = 0 if idle else 1
        self._expiration_time = now + timeout if idle else None

    def _acquire(self) -> None:
        if self.state not in (ContainerState.IDLE, ContainerState.BUSY):
            raise ValueError('Container non riutilizzabile')
        if self.requests_count >= self.function.max_concurrency:
            raise ValueError('Concorrenza massima raggiunta')
        self._requests_count += 1
        self._state = ContainerState.BUSY
        self._expiration_time = None

    def _complete(self, now: float, timeout: float) -> None:
        if self.state is not ContainerState.BUSY:
            raise ValueError('Nessuna richiesta attiva')
        self._requests_count -= 1
        if self.requests_count == 0:
            self._state = ContainerState.IDLE
            self._expiration_time = now + timeout

    def is_expired(self, now: float) -> bool:
        return (self.state is ContainerState.IDLE
                and self.expiration_time is not None and now > self.expiration_time)

    def _destroy(self) -> None:
        if self.state not in (ContainerState.IDLE, ContainerState.CREATING):
            raise ValueError('Distruzione ordinaria consentita solo per idle o creazione fallita')
        self._state = ContainerState.DESTROYED

    def __repr__(self) -> str:
        return f'DockerContainer({self.id!r}, {self.state.value}, requests={self.requests_count})'
