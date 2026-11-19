"""Richiesta di invocazione e risultato dell'ammissione sul nodo."""

from enum import Enum
from typing import TYPE_CHECKING

from .function import Function

if TYPE_CHECKING:
    from .container import DockerContainer
    from .node import Node


class RequestStatus(Enum):
    CREATED = 'created'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'


class Request:
    def __init__(self, function: Function, *, service_duration: float | None = None) -> None:
        self._function = function
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

    def _accept(self, node: 'Node', container: 'DockerContainer', warm: bool) -> None:
        self._ensure_new()
        self._node = node
        self._container = container
        self._warm = warm
        self._status = RequestStatus.ACCEPTED

    def _reject(self, reason: str) -> None:
        self._ensure_new()
        self._rejection_reason = reason
        self._status = RequestStatus.REJECTED
