"""Notifiche del ciclo di vita dei container."""

from abc import ABC
from dataclasses import dataclass, field
from logging import DEBUG, INFO
from typing import TYPE_CHECKING

from simulator.model.execution.container import ContainerState, DockerContainer
from simulator.model.execution.function import Function
from simulator.model.topology.node import Node

from .node import ResourceSnapshot
from .notification import Notification, NotificationType

if TYPE_CHECKING:
    from simulator.model.topology.region import Region
    from simulator.model.topology.ring import ArchitectureRing


@dataclass(frozen=True)
class _ContainerNotification(Notification, ABC):
    """Base comune per le notifiche legate al ciclo di vita di un container."""

    run_id: str
    sim_time_s: float
    ring: 'ArchitectureRing'
    node: Node
    container: DockerContainer
    region: 'Region | None' = field(default=None, kw_only=True)

    ### Domain references ###

    @property
    def function(self) -> Function:
        """Restituisce la funzione associata al container."""
        return self.container.function

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        region = f' regione={self.region.name}' if self.region is not None else ''
        return (
            f'[{self.sim_time_s:.6g} s] {self.type.value} '
            f'container={self.container.id} funzione={self.function.name} '
            f'nodo={self.node.id.key} ring={self.ring.id}{region}'
        )

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {
                'ring_id': self.ring.id,
                'region': self.region.name if self.region is not None else None,
                'node_id': self.node.id.key,
                'container_id': self.container.id,
                'function': self.function.name,
            },
        }


@dataclass(frozen=True)
class ContainerStateChanged(_ContainerNotification):
    """Segnala una transizione di stato del container con le prenotazioni aggiornate del nodo."""

    type = NotificationType.CONTAINER_STATE_CHANGED
    level = DEBUG

    previous: ContainerState | None = field(default=None, kw_only=True)
    state: ContainerState | None = field(default=None, kw_only=True)
    resources: ResourceSnapshot | None = field(default=None, kw_only=True)

    #####################
    ### Serialization ###
    #####################

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
                resources=self.resources.to_dict() if self.resources is not None else None,
            )
        return payload


@dataclass(frozen=True)
class ContainerStarted(ContainerStateChanged):
    """Un nuovo container è pronto per l'esecuzione o per il pool idle."""

    type = NotificationType.CONTAINER_STARTED
    level = INFO


@dataclass(frozen=True)
class ContainerStopped(ContainerStateChanged):
    """Un container già avviato è stato rimosso e ha liberato la RAM."""

    type = NotificationType.CONTAINER_STOPPED
    level = INFO
