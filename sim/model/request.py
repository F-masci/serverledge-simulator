"""Richiesta di invocazione e risultato dell'ammissione sul nodo."""

from collections.abc import Callable
from enum import Enum
import json
from typing import TYPE_CHECKING
from uuid import uuid4

from .function import Function
from ..utils.ensure import ensure_node, ensure_container

if TYPE_CHECKING:
    from .container import DockerContainer
    from .node import Node


class RequestStatus(Enum):
    """Stati di una richiesta durante ammissione ed esecuzione."""
    CREATED = 'created'
    QUEUED = 'queued'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    COMPLETED = 'completed'


class Request:
    """Rappresenta una singola invocazione di una funzione."""

    def __init__(
        self, function: Function, *, service_duration: float | None = None, arrival_time: float = 0.0,
    ) -> None:
        self._id = uuid4().hex
        self._function = function
        self._arrival_time = arrival_time
        self._service_duration = service_duration
        self._status = RequestStatus.CREATED
        self._node = None
        self._container = None
        self._warm = None
        self._rejection_reason = None

        ### Callback ###
        self.on_queued: Callable[[Request], None] | None = None
        self.on_accepted: Callable[[Request], None] | None = None
        self.on_rejected: Callable[[Request], None] | None = None
        self.on_completed: Callable[[Request], None] | None = None
        self.on_state_changed: Callable[[Request, RequestStatus, RequestStatus], None] | None = None

    @property
    def id(self) -> str:
        """Identificativo della singola invocazione."""
        return self._id

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

    def ensure_created(self) -> None:
        """Controlla che la richiesta sia ancora in attesa di ammissione."""
        if self.status is not RequestStatus.CREATED:
            raise ValueError('Richiesta già gestita')

    def enqueue(self, node: 'Node') -> None:
        """Assegna la richiesta alla FIFO del nodo."""
        self.ensure_created()

        # Conserva il nodo che potrà ammettere la richiesta dalla coda.
        self._node = node
        self._handle_change_status(RequestStatus.QUEUED, self.on_queued)

    def accept(self, node: 'Node', container: 'DockerContainer', warm: bool) -> None:
        """Ammette una nuova richiesta o una richiesta estratta dalla FIFO."""
        if self.status is not RequestStatus.QUEUED or self.node is not node:
            self.ensure_created()

        # I callback ricevono una richiesta con nodo, container e tipo di avvio già assegnati.
        self._node = node
        self._container = container
        self._warm = warm
        self._handle_change_status(RequestStatus.ACCEPTED, self.on_accepted)

    def reject(self, reason: str, *, node: 'Node | None' = None) -> None:
        """Registra il rifiuto definitivo e l'eventuale nodo destinatario."""
        self.ensure_created()

        # Un rifiuto di routing può avvenire prima di individuare un nodo.
        self._node = node
        self._rejection_reason = reason
        self._handle_change_status(RequestStatus.REJECTED, self.on_rejected)

    def complete(self, now: float) -> None:
        """Conclude l'invocazione e aggiorna il container sul nodo assegnato."""
        node = ensure_node(self)
        container = ensure_container(self)
        if self.status is not RequestStatus.ACCEPTED:
            raise ValueError('Solo una richiesta accettata può essere completata')

        # Il nodo libera le risorse e prova ad avviare le richieste in FIFO.
        node.complete(container, now)
        self._handle_change_status(RequestStatus.COMPLETED, self.on_completed)

    def _handle_change_status(self, status: RequestStatus, callback: Callable[[Request], None] | None) -> None:
        """Aggiorna lo stato prima di invocare i callback."""
        previous_status = self._status
        self._status = status

        # Notifica prima la transizione specifica, poi il cambio di stato generico.
        if callback is not None:
            callback(self)
        if self.on_state_changed is not None:
            self.on_state_changed(self, previous_status, status)


    def __str__(self) -> str:
        """Restituisce una descrizione leggibile della richiesta."""
        fields = self.to_dict()
        details = ', '.join(f'{name}={value}' for name, value in fields.items() if value is not None)
        return f'Request({details})'

    def to_dict(self) -> dict:
        """Restituisce lo stato della richiesta con i riferimenti espressi come ID."""
        node = ensure_node(self)
        container = ensure_container(self)
        return {
            'id': self.id,
            'function': self.function.name,
            'status': self.status.value,
            'arrival_time_s': self.arrival_time,
            'service_duration_s': self.service_duration,
            'node_id': node.id.key if node is not None else None,
            'container_id': container.id if container is not None else None,
            'warm': self.warm,
            'rejection_reason': self.rejection_reason,
        }

    def to_json(self) -> str:
        """Serializza la richiesta per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
