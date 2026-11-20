"""Richiesta di invocazione e risultato dell'ammissione sul nodo."""

from collections.abc import Callable
from enum import Enum
import json
from typing import TYPE_CHECKING
from uuid import uuid4

from .function import Function
from ..utils.ensure import ensure_container

if TYPE_CHECKING:
    from .container import DockerContainer
    from .node import Node


class RequestStatus(Enum):
    """Stati di una richiesta durante ammissione ed esecuzione."""
    CREATED = 'created'
    IN_TRANSIT = 'in_transit'
    QUEUED = 'queued'
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    RESPONDING = 'responding'
    COMPLETED = 'completed'


class Request:
    """Rappresenta una singola invocazione di una funzione."""

    def __init__(
        self, function: Function, *, service_duration: float | None = None, arrival_time: float = 0.0,
        can_offload: bool = False,
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
        self._allow_offloading = can_offload
        self._offloaded_from: Node | None = None
        self._entry_region: str | None = None
        self._region: str | None = None
        self._region_arrival_time = arrival_time
        self._return_delay = 0.0

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

    @property
    def can_offload(self) -> bool:
        return self._allow_offloading and self._offloaded_from is None

    @property
    def offloaded_from(self) -> 'Node | None':
        return self._offloaded_from

    @property
    def entry_region(self) -> str | None:
        return self._entry_region

    @property
    def region(self) -> str | None:
        return self._region

    @property
    def region_arrival_time(self) -> float:
        return self._region_arrival_time

    @property
    def return_delay(self) -> float:
        return self._return_delay

    def enter_region(self, region: str, now: float) -> None:
        if self.status is not RequestStatus.IN_TRANSIT:
            self.ensure_created()
        self._region = region
        if self._entry_region is None:
            self._entry_region = region
        self._region_arrival_time = now
        if self.status is RequestStatus.IN_TRANSIT:
            self._handle_change_status(RequestStatus.CREATED, None)

    def start_transfer(self, region: str, return_delay: float) -> None:
        self.ensure_created()
        self._region = region
        self._return_delay = return_delay
        self._handle_change_status(RequestStatus.IN_TRANSIT, None)

    def mark_offloaded(self, source: 'Node') -> None:
        self.ensure_created()
        if not self.can_offload:
            raise ValueError('Offloading non consentito per questa richiesta')
        self._offloaded_from = source

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

    def accept_without_container(self, service_time: float) -> None:
        self.ensure_created()
        self._service_duration = service_time
        self._handle_change_status(RequestStatus.ACCEPTED, self.on_accepted)

    def complete(self, now: float) -> None:
        """Conclude un'invocazione con risposta immediata."""
        self._release_execution(now)
        self._handle_change_status(RequestStatus.COMPLETED, self.on_completed)

    def begin_response(self, now: float) -> None:
        """Libera il container prima del rientro della risposta."""
        self._release_execution(now)
        self._handle_change_status(RequestStatus.RESPONDING, None)

    def finish_response(self) -> None:
        if self.status is not RequestStatus.RESPONDING:
            raise ValueError('La richiesta non ha una risposta in transito')
        self._handle_change_status(RequestStatus.COMPLETED, self.on_completed)

    def _release_execution(self, now: float) -> None:
        if self.status is not RequestStatus.ACCEPTED:
            raise ValueError('Solo una richiesta accettata può essere completata')

        if self.node is not None:
            self.node.complete(ensure_container(self), now)

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
        node = self.node
        container = self.container
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
            'entry_region': self.entry_region,
            'region': self.region,
            'offloaded_from_node_id': self.offloaded_from.id.key if self.offloaded_from is not None else None,
        }

    def to_json(self) -> str:
        """Serializza la richiesta per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
