"""Notifiche di arrivo, rifiuto e completamento delle richieste."""
from abc import ABC
from dataclasses import asdict, dataclass, field
from logging import INFO

from .notification import Notification, NotificationType


@dataclass(frozen=True)
class _RequestNotification(Notification, ABC):
    run_id: str
    sim_time_s: float
    request_id: str
    function: str
    region: str | None = field(default=None, kw_only=True)

    def __str__(self) -> str:
        data = self.to_dict()['data']
        details = ' '.join(f'{name}={value}' for name, value in data.items() if value is not None)
        return f'[{self.sim_time_s:.6g} s] {self.type.value} {details}'

    def to_dict(self) -> dict:
        data = asdict(self)
        return {
            'type': self.type.value,
            'run_id': data.pop('run_id'),
            'sim_time_s': data.pop('sim_time_s'),
            'data': data,
        }


@dataclass(frozen=True)
class RequestArrived(_RequestNotification):
    """La richiesta è arrivata e deve ancora essere inoltrata."""

    type = NotificationType.REQUEST_ARRIVED
    level = INFO


@dataclass(frozen=True)
class RequestAccepted(_RequestNotification):
    """La richiesta è stata ammessa nella regione indicata."""

    type = NotificationType.REQUEST_ACCEPTED
    level = INFO

    node_id: str | None


@dataclass(frozen=True)
class RequestRejected(_RequestNotification):
    """La richiesta è stata rifiutata definitivamente."""

    type = NotificationType.REQUEST_REJECTED
    level = INFO

    reason: str
    node_id: str | None


@dataclass(frozen=True)
class RequestCompleted(_RequestNotification):
    """La risposta dell'invocazione è stata consegnata."""

    type = NotificationType.REQUEST_COMPLETED
    level = INFO

    node_id: str | None
    container_id: str | None
    response_time_s: float
