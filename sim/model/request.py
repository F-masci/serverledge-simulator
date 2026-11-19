"""Richiesta di invocazione e risultato dell'ammissione sul nodo."""

from enum import Enum
from typing import TYPE_CHECKING

from .function import Function

if TYPE_CHECKING:
    from .container import DockerContainer
    from .node import Node


class RequestStatus(Enum):
    CREATED = 'created'
    QUEUED = 'queued'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    COMPLETED = 'completed'


class Request:
    def __init__(
        self, function: Function, *, service_duration: float | None = None, arrival_time: float = 0.0,
    ) -> None:
        self._function = function
        self._arrival_time = arrival_time
        self._service_duration = service_duration
        self._status = RequestStatus.CREATED
        self._node = None
        self._container = None
        self._warm = None
        self._rejection_reason = None

    @property
    def function(self) -> Function:
        return self._function

    @property
    def service_duration(self) -> float | None:
        """Durata di servizio richiesta, in secondi simulati."""
        return self._service_duration

    @property
    def arrival_time(self) -> float:
        return self._arrival_time

    @property
    def status(self) -> RequestStatus:
        return self._status

    @property
    def node(self) -> 'Node | None':
        return self._node

    @property
    def container(self) -> 'DockerContainer | None':
        return self._container

    @property
    def warm(self) -> bool | None:
        return self._warm

    @property
    def rejection_reason(self) -> str | None:
        return self._rejection_reason

    def _ensure_new(self) -> None:
        if self.status is not RequestStatus.CREATED:
            raise ValueError('Richiesta già gestita')

    def _enqueue(self, node: 'Node') -> None:
        self._ensure_new()
        self._node = node
        self._status = RequestStatus.QUEUED

    def _accept(self, node: 'Node', container: 'DockerContainer', warm: bool) -> None:
        if self.status is not RequestStatus.QUEUED or self.node is not node:
            self._ensure_new()
        self._node = node
        self._container = container
        self._warm = warm
        self._status = RequestStatus.ACCEPTED

    def _reject(self, reason: str, *, node: 'Node | None' = None) -> None:
        self._ensure_new()
        self._node = node
        self._rejection_reason = reason
        self._status = RequestStatus.REJECTED

    def complete(self, now: float) -> None:
        """Conclude l'invocazione e aggiorna il container sul nodo assegnato."""
        if self.status is not RequestStatus.ACCEPTED or self.node is None or self.container is None:
            raise ValueError('Solo una richiesta accettata può essere completata')
        self.node.complete(self.container, now)
        self._status = RequestStatus.COMPLETED
