"""Notifiche del ciclo di vita delle richieste (arrivo, accodamento, inoltro, esito)."""

from abc import ABC
from dataclasses import dataclass, field
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, ClassVar

from simulator.model.execution.function import Function
from simulator.model.execution.request import Request

from .notification import Notification, NotificationType

if TYPE_CHECKING:
    from simulator.model.execution.container import DockerContainer
    from simulator.model.topology.node import Node
    from simulator.model.topology.region import Region
    from simulator.model.topology.ring import ArchitectureRing


_STR_LABELS: dict[str, str] = {
    'request_id': 'richiesta',
    'function': 'funzione',
    'region': 'regione',
    'ring_id': 'anello',
    'node_id': 'nodo',
    'source_region': 'origine',
    'warm': 'warm',
    'queue_wait_s': 'attesa_coda_s',
    'reason': 'motivo',
    'container_id': 'container',
    'response_time_s': 'tempo_risposta_s',
    'service_time_s': 'tempo_servizio_s',
}


@dataclass(frozen=True)
class _RequestNotification(Notification, ABC):
    """Base comune per le notifiche associate a una richiesta di invocazione."""

    run_id: str
    sim_time_s: float
    request: Request
    region: 'Region | None' = field(default=None, kw_only=True)
    ring: 'ArchitectureRing | None' = field(default=None, kw_only=True)

    ### Domain references ###

    @property
    def function(self) -> Function:
        """Restituisce la funzione invocata dalla richiesta."""
        return self.request.function

    #####################
    ### Serialization ###
    #####################

    def _data(self) -> dict:
        """Costruisce il payload serializzabile di base esprimendo i riferimenti come ID."""
        return {
            'request_id': self.request.id,
            'function': self.function.name,
            'region': self.region.name if self.region is not None else None,
            'ring_id': self.ring.id if self.ring is not None else None,
        }

    def __str__(self) -> str:
        data = self.to_dict()['data']
        details = ' '.join(
            f'{_STR_LABELS.get(name, name)}={value}'
            for name, value in data.items()
            if value is not None
        )
        return f'[{self.sim_time_s:.6g} s] {self.type.value} {details}'

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': self._data(),
        }


@dataclass(frozen=True)
class RequestArrived(_RequestNotification):
    """La richiesta è arrivata nella regione di ingresso e deve ancora essere instradata."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_ARRIVED
    level: ClassVar[int] = INFO


@dataclass(frozen=True)
class RequestQueued(_RequestNotification):
    """La richiesta è in attesa nella coda FIFO del nodo."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_QUEUED
    level: ClassVar[int] = DEBUG

    node: 'Node'

    def _data(self) -> dict:
        return {**super()._data(), 'node_id': self.node.id.key}


@dataclass(frozen=True)
class RequestDequeued(_RequestNotification):
    """La richiesta è uscita dalla coda FIFO del nodo."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_DEQUEUED
    level: ClassVar[int] = DEBUG

    node: 'Node'

    def _data(self) -> dict:
        return {**super()._data(), 'node_id': self.node.id.key}


@dataclass(frozen=True)
class RequestForwarded(_RequestNotification):
    """La richiesta parte dalla regione di origine verso la regione di destinazione."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_FORWARDED
    level: ClassVar[int] = DEBUG

    source_region: 'Region'

    def _data(self) -> dict:
        return {**super()._data(), 'source_region': self.source_region.name}


@dataclass(frozen=True)
class RequestDelivered(_RequestNotification):
    """La richiesta inoltrata arriva nella regione di destinazione."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_DELIVERED
    level: ClassVar[int] = DEBUG


@dataclass(frozen=True)
class RequestAccepted(_RequestNotification):
    """La richiesta è stata ammessa nella regione indicata (su container o regione sintetica)."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_ACCEPTED
    level: ClassVar[int] = INFO

    node: 'Node | None'
    warm: bool | None = field(default=None, kw_only=True)
    queue_wait_s: float | None = field(default=None, kw_only=True)

    def _data(self) -> dict:
        return {
            **super()._data(),
            'node_id': self.node.id.key if self.node is not None else None,
            'warm': self.warm,
            'queue_wait_s': self.queue_wait_s,
        }


@dataclass(frozen=True)
class RequestRejectionReturning(_RequestNotification):
    """La richiesta è stata rifiutata da una regione remota e il rifiuto è in ritorno."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_REJECTION_RETURNING
    level: ClassVar[int] = DEBUG

    reason: str
    node: 'Node | None'

    def _data(self) -> dict:
        return {
            **super()._data(),
            'reason': self.reason,
            'node_id': self.node.id.key if self.node is not None else None,
        }


@dataclass(frozen=True)
class RequestRejected(_RequestNotification):
    """La richiesta è stata rifiutata definitivamente durante il routing o l'ammissione."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_REJECTED
    level: ClassVar[int] = INFO

    reason: str
    node: 'Node | None'

    def _data(self) -> dict:
        return {
            **super()._data(),
            'reason': self.reason,
            'node_id': self.node.id.key if self.node is not None else None,
        }


@dataclass(frozen=True)
class RequestCompleted(_RequestNotification):
    """La risposta dell'invocazione è stata consegnata al client."""

    type: ClassVar[NotificationType] = NotificationType.REQUEST_COMPLETED
    level: ClassVar[int] = INFO

    node: 'Node | None'
    container: 'DockerContainer | None'
    response_time_s: float
    service_time_s: float | None = field(default=None, kw_only=True)

    def _data(self) -> dict:
        return {
            **super()._data(),
            'node_id': self.node.id.key if self.node is not None else None,
            'container_id': self.container.id if self.container is not None else None,
            'response_time_s': self.response_time_s,
            'service_time_s': self.service_time_s,
        }
