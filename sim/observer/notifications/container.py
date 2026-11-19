"""Notifiche del ciclo di vita dei container."""

from dataclasses import dataclass

from .notification import Notification, NotificationType


@dataclass(frozen=True)
class _ContainerNotification(Notification):
    run_id: str
    sim_time_s: float
    ring_id: str
    node_id: str
    container_id: str
    function: str

    def __str__(self) -> str:
        return (
            f'[{self.sim_time_s:.6g} s] {self.type.value} '
            f'container={self.container_id} funzione={self.function} '
            f'nodo={self.node_id} ring={self.ring_id}'
        )

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {
                'ring_id': self.ring_id,
                'node_id': self.node_id,
                'container_id': self.container_id,
                'function': self.function,
            },
        }


@dataclass(frozen=True)
class ContainerStarted(_ContainerNotification):
    """Un nuovo container è pronto per l'esecuzione o il pool idle."""

    type = NotificationType.CONTAINER_STARTED


@dataclass(frozen=True)
class ContainerStopped(_ContainerNotification):
    """Un container già avviato è stato rimosso e ha liberato la RAM."""

    type = NotificationType.CONTAINER_STOPPED
