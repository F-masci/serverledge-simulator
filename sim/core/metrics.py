"""Metriche delle invocazioni e delle prenotazioni Serverledge."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from sim.model.infrastructure import Infrastructure
from sim.model.node import Node
from sim.model.request import Request, RequestStatus
from sim.utils.welford import TimedWelford, Welford


@dataclass
class _InvocationMetrics:
    accepted: int = 0
    completed: int = 0
    cold_starts: int = 0
    warm_starts: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    response_time: Welford = field(default_factory=Welford)
    service_time: Welford = field(default_factory=Welford)

    @property
    def arrivals(self) -> int:
        return self.accepted + self.rejections.total()

    def record_admission(self, request: Request) -> None:
        if request.status is RequestStatus.ACCEPTED:
            self.accepted += 1
            if request.warm:
                self.warm_starts += 1
            else:
                self.cold_starts += 1
        else:
            self.rejections[request.rejection_reason or 'unknown'] += 1

    def record_completion(self, response_time: float, service_time: float) -> None:
        self.completed += 1
        self.response_time.add(response_time)
        self.service_time.add(service_time)

    def snapshot(self) -> dict:
        return {
            'arrivals': self.arrivals,
            'accepted': self.accepted,
            'rejected': self.rejections.total(),
            'completed': self.completed,
            'pending': self.accepted - self.completed,
            'cold_starts': self.cold_starts,
            'warm_starts': self.warm_starts,
            'rejections_by_reason': dict(self.rejections),
            'mean_response_time_s': self.response_time.mean,
            'response_time_variance_s2': self.response_time.variance,
            'mean_service_time_s': self.service_time.mean,
            'service_time_variance_s2': self.service_time.variance,
        }


class _NodeMetrics:
    def __init__(self, node: Node, now: float) -> None:
        self.invocations = _InvocationMetrics()
        self.by_function: dict[str, _InvocationMetrics] = {}
        self.resources = {
            name: TimedWelford(now, value) for name, value in self._reservations(node).items()
        }

    @staticmethod
    def _reservations(node: Node) -> dict[str, float]:
        resources = node.resources
        return {
            'used_cpus': resources.used_cpus,
            'busy_memory_mb': resources.busy_pool_used_mem,
            'warm_memory_mb': resources.warm_pool_used_mem,
        }

    def observe(self, node: Node, now: float) -> None:
        for name, value in self._reservations(node).items():
            self.resources[name].observe(now, value)

    def snapshot(self, node: Node, now: float) -> dict:
        time_weighted_resources = {}
        for name, accumulator in self.resources.items():
            moments = accumulator.snapshot(now)
            time_weighted_resources[name] = {
                'observed_time_s': moments.weight,
                'mean': moments.mean,
                'variance': moments.variance,
            }
        return {
            **self._reservations(node),
            'total_cpus': node.resources.total_cpus,
            'total_memory_mb': node.resources.total_memory,
            'invocations': self.invocations.snapshot(),
            'by_function': {name: stats.snapshot() for name, stats in self.by_function.items()},
            'time_weighted_resources': time_weighted_resources,
        }


class SimulationMetrics:
    """Raccoglie statistiche online e contatori globali, per funzione e per nodo."""

    def __init__(self) -> None:
        self._total = _InvocationMetrics()
        self._by_function: dict[str, _InvocationMetrics] = {}
        self._by_node: dict[Node, _NodeMetrics] = {}

    @property
    def arrivals(self) -> int:
        return self._total.arrivals

    def register_node(self, node: Node, now: float) -> None:
        """Fissa lo stato iniziale delle prenotazioni del nodo."""
        if node not in self._by_node:
            self._by_node[node] = _NodeMetrics(node, now)

    def _invocations_for(self, request: Request, now: float) -> Iterator[_InvocationMetrics]:
        name = request.function.name
        if name not in self._by_function:
            self._by_function[name] = _InvocationMetrics()
        yield self._total
        yield self._by_function[name]
        node = request.node
        if node is not None:
            self.register_node(node, now)
            stats = self._by_node[node]
            if name not in stats.by_function:
                stats.by_function[name] = _InvocationMetrics()
            stats.observe(node, now)
            yield stats.invocations
            yield stats.by_function[name]

    def record_admission(self, request: Request, now: float) -> None:
        for stats in self._invocations_for(request, now):
            stats.record_admission(request)

    def record_completion(self, request: Request, now: float, arrival_time: float) -> None:
        service_time = request.service_duration
        assert service_time is not None  # Le richieste simulate hanno una durata campionata.
        for stats in self._invocations_for(request, now):
            stats.record_completion(now - arrival_time, service_time)

    def snapshot(self, now: float, infrastructure: Infrastructure) -> dict:
        """Restituisce e separa le metriche dei nodi per ring e nome."""
        nodes_by_ring = {}
        for ring_id, ring in infrastructure.rings.items():
            nodes = {}
            for node in ring:
                stats = self._by_node.get(node)
                if stats is None:
                    stats = _NodeMetrics(node, now)
                nodes[node.id.key] = stats.snapshot(node, now)
            nodes_by_ring[ring_id] = nodes
        return {
            'time_s': now,
            'total': self._total.snapshot(),
            'by_function': {name: stats.snapshot() for name, stats in self._by_function.items()},
            'nodes_by_ring': nodes_by_ring,
        }
