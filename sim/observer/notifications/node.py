"""Notifiche di registrazione dei nodi e stato delle prenotazioni."""

from dataclasses import asdict, dataclass
from logging import DEBUG

from sim.model.container import ContainerState
from sim.model.node import Node
from .notification import Notification, NotificationType


@dataclass(frozen=True)
class ResourceSnapshot:
    used_cpus: float
    busy_memory_mb: int
    warm_memory_mb: int
    total_cpus: float
    total_memory_mb: int

    @classmethod
    def from_node(cls, node: Node) -> 'ResourceSnapshot':
        resources = node.resources
        return cls(
            resources.used_cpus, resources.busy_pool_used_mem, resources.warm_pool_used_mem,
            resources.total_cpus, resources.total_memory,
        )


@dataclass(frozen=True)
class ContainerSnapshot:
    container_id: str
    function: str
    state: ContainerState


@dataclass(frozen=True)
class NodeRegistered(Notification):
    """Il nodo entra nella replica con le prenotazioni e i container correnti."""

    type = NotificationType.NODE_REGISTERED
    level = DEBUG

    run_id: str
    sim_time_s: float
    region: str
    ring_id: str
    node_id: str
    resources: ResourceSnapshot
    containers: tuple[ContainerSnapshot, ...]

    def __str__(self) -> str:
        return f'[{self.sim_time_s:.6g} s] {self.type.value} nodo={self.node_id} ring={self.ring_id} regione={self.region}'

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {
                'region': self.region,
                'ring_id': self.ring_id,
                'node_id': self.node_id,
                'resources': asdict(self.resources),
                'containers': [
                    {'container_id': container.container_id, 'function': container.function, 'state': container.state.value}
                    for container in self.containers
                ],
            },
        }
