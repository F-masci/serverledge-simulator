"""Notifiche di registrazione dei nodi e snapshot delle prenotazioni."""

from dataclasses import asdict, dataclass
import json
from logging import DEBUG
from typing import TYPE_CHECKING

from sim.model.container import ContainerState, DockerContainer
from sim.model.node import Node
from .notification import Notification, NotificationType

if TYPE_CHECKING:
    from sim.model.region import Region
    from sim.model.ring import ArchitectureRing


@dataclass(frozen=True)
class ResourceSnapshot:
    """Fotografia immutabile delle prenotazioni di CPU e RAM di un nodo in un istante."""

    used_cpus: float
    busy_memory_mb: int
    warm_memory_mb: int
    total_cpus: float
    total_memory_mb: int

    @classmethod
    def from_node(cls, node: Node) -> 'ResourceSnapshot':
        """Cattura le prenotazioni correnti del nodo indicato."""
        resources = node.resources
        return cls(
            resources.used_cpus, resources.busy_pool_used_mem, resources.warm_pool_used_mem,
            resources.total_cpus, resources.total_memory,
        )

    #####################
    ### Serialization ###
    #####################

    def to_dict(self) -> dict:
        """Restituisce lo snapshot delle risorse in formato serializzabile."""
        return asdict(self)

    def to_json(self) -> str:
        """Serializza lo snapshot delle risorse in JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class ContainerSnapshot:
    """Fotografia di un container e del suo stato all'istante della registrazione del nodo."""

    container: DockerContainer
    state: ContainerState

    #####################
    ### Serialization ###
    #####################

    def to_dict(self) -> dict:
        """Restituisce lo snapshot del container con i riferimenti espressi come ID."""
        return {
            'container_id': self.container.id,
            'function': self.container.function.name,
            'state': self.state.value,
        }

    def to_json(self) -> str:
        """Serializza lo snapshot del container in JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class NodeRegistered(Notification):
    """Notifica l'ingresso di un nodo nella replica con prenotazioni e container correnti."""

    type = NotificationType.NODE_REGISTERED
    level = DEBUG

    run_id: str
    sim_time_s: float
    region: 'Region'
    ring: 'ArchitectureRing'
    node: Node
    resources: ResourceSnapshot
    containers: tuple[ContainerSnapshot, ...]

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        return (
            f'[{self.sim_time_s:.6g} s] {self.type.value} '
            f'nodo={self.node.id.key} ring={self.ring.id} regione={self.region.name}'
        )

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {
                'region': self.region.name,
                'ring_id': self.ring.id,
                'node_id': self.node.id.key,
                'resources': self.resources.to_dict(),
                'containers': [snapshot.to_dict() for snapshot in self.containers],
            },
        }
