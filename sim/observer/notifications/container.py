"""Notifiche del ciclo di vita dei container."""

from dataclasses import asdict, dataclass, field
from logging import DEBUG, INFO

from sim.model.container import ContainerState
from .node import ResourceSnapshot
from .notification import Notification, NotificationType


@dataclass(frozen=True)
class _ContainerNotification(Notification):
    run_id: str
    sim_time_s: float
    ring_id: str
    node_id: str
    container_id: str
    function: str
    region: str | None = field(default=None, kw_only=True)

    def __str__(self) -> str:
        region = f' regione={self.region}' if self.region is not None else ''
        return (
            f'[{self.sim_time_s:.6g} s] {self.type.value} '
            f'container={self.container_id} funzione={self.function} '
            f'nodo={self.node_id} ring={self.ring_id}{region}'
        )

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {
                'ring_id': self.ring_id,
                'region': self.region,
                'node_id': self.node_id,
                'container_id': self.container_id,
                'function': self.function,
            },
        }


@dataclass(frozen=True)
class ContainerStateChanged(_ContainerNotification):
    """Una transizione del container con le prenotazioni aggiornate del nodo."""

    type = NotificationType.CONTAINER_STATE_CHANGED
    level = DEBUG

    previous: ContainerState | None = field(default=None, kw_only=True)
    state: ContainerState | None = field(default=None, kw_only=True)
    resources: ResourceSnapshot | None = field(default=None, kw_only=True)

    def __str__(self) -> str:
        text = super().__str__()
        if self.state is None:
            return text
        previous = self.previous.value if self.previous is not None else 'none'
        return f'{text} stato={previous}→{self.state.value}'

    def to_dict(self) -> dict:
        payload = super().to_dict()
        if self.state is not None:
            payload['data'].update(
                previous=self.previous.value if self.previous is not None else None,
                state=self.state.value,
                resources=asdict(self.resources) if self.resources is not None else None,
            )
        return payload


@dataclass(frozen=True)
class ContainerStarted(ContainerStateChanged):
    """Un nuovo container è pronto per l'esecuzione o il pool idle."""

    type = NotificationType.CONTAINER_STARTED
    level = INFO


@dataclass(frozen=True)
class ContainerStopped(ContainerStateChanged):
    """Un container già avviato è stato rimosso e ha liberato la RAM."""

    type = NotificationType.CONTAINER_STOPPED
    level = INFO
