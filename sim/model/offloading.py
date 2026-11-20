"""Politiche di scheduling delle invocazioni di Serverledge."""

from dataclasses import dataclass
from enum import StrEnum
from itertools import islice

from .node import Node
from .request import Request


class SchedulingPolicy(StrEnum):
    DEFAULT = 'default'
    EDGE_ONLY = 'edgeonly'
    EDGE_CLOUD = 'edgecloud'
    CLOUD_ONLY = 'cloudonly'

    @classmethod
    def _missing_(cls, value: object) -> 'SchedulingPolicy | None':
        # CreateSchedulingPolicy in Go ricade sulla policy locale per nomi sconosciuti.
        return cls.DEFAULT if isinstance(value, str) else None


@dataclass(frozen=True)
class CloudLoadBalancer:
    area: str


@dataclass(frozen=True)
class RemoteRegion:
    name: str


@dataclass(frozen=True)
class NodeScheduling:
    policy: SchedulingPolicy = SchedulingPolicy.DEFAULT
    fallback_local: bool = False
    remote_area: str | None = None
    nearest_neighbors: tuple[Node, ...] | None = None
    remote_region: str | None = None


_DEFAULT_SCHEDULING = NodeScheduling()


class OffloadingScheduler:
    """Esegue la policy locale o restituisce l'unico destinatario remoto scelto."""

    def __init__(self, *, cache_validity: int = 60, load_balancer_areas: tuple[str, ...] = ()) -> None:
        self._cache_validity = cache_validity
        self._load_balancer_areas = frozenset(load_balancer_areas)
        self._settings: dict[Node, NodeScheduling] = {}
        self._nearby: dict[Node, tuple[tuple[Node, int], ...]] = {}
        self._edge_cache: dict[tuple[Node, str], tuple[Node, float]] = {}
        self._remote: dict[Node, Node | CloudLoadBalancer] = {}

    def configure(self, node: Node, settings: NodeScheduling) -> None:
        self._settings[node] = settings

    def refresh_neighbors(self, nodes: tuple[Node, ...]) -> None:
        registered = set(nodes)
        for node, settings in self._settings.items():
            if node not in registered or settings.policy is not SchedulingPolicy.EDGE_ONLY:
                continue
            neighbors = settings.nearest_neighbors
            if neighbors is None:
                peers = (peer for peer in nodes if peer is not node and peer.id.area == node.id.area)
                neighbors = tuple(islice(peers, 2))
            self._nearby[node] = tuple(
                (peer, peer.resources.available_memory) for peer in neighbors
                if peer in registered and peer is not node and peer.id.area == node.id.area
            )

    def refresh_remote_targets(self, nodes: tuple[Node, ...]) -> None:
        # GetOneNodeInArea legge la prima chiave del prefisso etcd (machine tag, poi ID).
        first_by_area: dict[str, Node] = {}
        for node in sorted(nodes, key=lambda n: f'{n.id.machine_tag}/{n.id.key}'):
            first_by_area.setdefault(node.id.area, node)

        for node, settings in self._settings.items():
            area = settings.remote_area
            if area is None:
                continue
            if area in self._load_balancer_areas:
                self._remote[node] = CloudLoadBalancer(area)
            elif (target := first_by_area.get(area)) is not None and target is not node:
                self._remote[node] = target

    def schedule(self, request: Request, node: Node, now: float) -> Node | CloudLoadBalancer | RemoteRegion | None:
        """Ammette o rifiuta la richiesta, oppure la destina a un hop remoto."""
        settings = self._settings.get(node, _DEFAULT_SCHEDULING)
        policy = settings.policy
        supports_arch = request.function.supports_arch(node.id.arch)
        if policy is SchedulingPolicy.DEFAULT:
            if supports_arch:
                node.submit(request)
            else:
                request.reject('out_of_resources', node=node)
            return None

        if policy is SchedulingPolicy.EDGE_ONLY:
            if request.can_offload:
                target = self._edge_target(request, node, now)
                if target is not None:
                    return target
                # In Go NoNeighbors (snapshot nil) e NoSuitableNode sono due esiti distinti.
                try_local = node in self._nearby and settings.fallback_local
            else:
                try_local = True
            if try_local and supports_arch and node.submit(request, reject_on_failure=False):
                return None
        else:
            if policy is SchedulingPolicy.EDGE_CLOUD and supports_arch:
                if node.submit(request, reject_on_failure=False):
                    return None
            if request.can_offload:
                if settings.remote_region is not None:
                    return RemoteRegion(settings.remote_region)
                target = self._remote.get(node)
                if isinstance(target, CloudLoadBalancer):
                    return target
                if target is not None and request.function.supports_arch(target.id.arch):
                    return target

        request.reject('out_of_resources', node=node)
        return None

    def _edge_target(self, request: Request, node: Node, now: float) -> Node | None:
        key = (node, request.function.name)
        cached = self._edge_cache.get(key)
        if cached is not None and now < cached[1]:
            return cached[0]

        function = request.function
        suitable = (
            (peer, memory) for peer, memory in self._nearby.get(node, ())
            if memory > 0 and function.supports_arch(peer.id.arch)
        )
        best = max(suitable, key=lambda candidate: candidate[1], default=None)
        if best is None:
            return None
        target = best[0]
        self._edge_cache[key] = (target, now + self._cache_validity)
        return target
