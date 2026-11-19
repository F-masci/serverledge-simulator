"""Metriche delle invocazioni e delle prenotazioni Serverledge."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from sim.model.container import ContainerState, DockerContainer
from sim.model.infrastructure import Infrastructure
from sim.model.node import Node
from sim.model.request import Request, RequestStatus
from sim.utils.welford import TimedWelford, Welford


@dataclass
class _InvocationMetrics:
    waiting: int = 0
    queued: int = 0
    queue_wait: Welford = field(default_factory=Welford)
    accepted: int = 0
    completed: int = 0
    cold_starts: int = 0
    warm_starts: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    response_time: Welford = field(default_factory=Welford)
    service_time: Welford = field(default_factory=Welford)

    @property
    def arrivals(self) -> int:
        return self.waiting + self.accepted + self.rejections.total()

    def record_submission(self, request: Request) -> None:
        if request.status is RequestStatus.QUEUED:
            self.waiting += 1
            self.queued += 1
        elif request.status is RequestStatus.ACCEPTED:
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
            'pending': self.waiting + self.accepted - self.completed,
            'waiting': self.waiting,
            'queued': self.queued,
            'mean_queue_wait_s': self.queue_wait.mean,
            'cold_starts': self.cold_starts,
            'warm_starts': self.warm_starts,
            'rejections_by_reason': dict(self.rejections),
            'mean_response_time_s': self.response_time.mean,
            'response_time_variance_s2': self.response_time.variance,
            'mean_service_time_s': self.service_time.mean,
            'service_time_variance_s2': self.service_time.variance,
        }


_LIVE_STATES = (ContainerState.CREATING, ContainerState.BUSY, ContainerState.IDLE)


def _timed_snapshot(accumulator: TimedWelford, now: float) -> dict:
    moments = accumulator.snapshot(now)
    return {'observed_time_s': moments.weight, 'mean': moments.mean, 'variance': moments.variance}


class _ContainerMetrics:
    def __init__(self) -> None:
        self.counts = dict.fromkeys(_LIVE_STATES, 0)
        self.time_weighted = {state: TimedWelford(0, 0) for state in _LIVE_STATES}
        self.creations_started = 0
        self.created = 0
        self.creation_failed = 0
        self.destroyed = 0
        self.creation_time = Welford()

    def observe(self, now: float, state: ContainerState, delta: int) -> None:
        self.counts[state] += delta
        self.time_weighted[state].observe(now, self.counts[state])

    def record_transition(
        self, now: float, previous: ContainerState | None, state: ContainerState,
        creation_time: float | None,
    ) -> None:
        if previous is not None:
            self.observe(now, previous, -1)
        if state in self.counts:
            self.observe(now, state, 1)
        if previous is None:
            self.creations_started += 1
        elif previous is ContainerState.CREATING:
            if state is ContainerState.DESTROYED:
                self.creation_failed += 1
            else:
                self.created += 1
                if creation_time is not None:
                    self.creation_time.add(creation_time)
        elif state is ContainerState.DESTROYED:
            self.destroyed += 1

    def snapshot(self, now: float) -> dict:
        return {
            **{state.value: count for state, count in self.counts.items()},
            'creations_started': self.creations_started,
            'created': self.created,
            'creation_failed': self.creation_failed,
            'destroyed': self.destroyed,
            'mean_creation_time_s': self.creation_time.mean,
            'creation_time_variance_s2': self.creation_time.variance,
            'time_weighted_counts': {
                state.value: _timed_snapshot(accumulator, now)
                for state, accumulator in self.time_weighted.items()
            },
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
        return {
            **self._reservations(node),
            'total_cpus': node.resources.total_cpus,
            'total_memory_mb': node.resources.total_memory,
            'invocations': self.invocations.snapshot(),
            'by_function': {name: stats.snapshot() for name, stats in self.by_function.items()},
            'time_weighted_resources': {
                name: _timed_snapshot(accumulator, now) for name, accumulator in self.resources.items()
            },
        }


class SimulationMetrics:
    """Raccoglie statistiche online e contatori globali, per funzione e per nodo."""

    def __init__(self) -> None:
        self._total = _InvocationMetrics()
        self._by_function: dict[str, _InvocationMetrics] = {}
        self._by_node: dict[Node, _NodeMetrics] = {}
        self._containers: dict[tuple[Node | None, str | None], _ContainerMetrics] = {}
        self._container_functions_by_node: dict[Node, set[str]] = {}
        self._container_functions: set[str] = set()
        self._creation_started_at: dict[DockerContainer, float] = {}

    def __str__(self) -> str:
        """Riepilogo globale leggibile delle invocazioni e dei container."""
        totals = self._total.snapshot()
        containers = self._containers.get((None, None))
        if containers is None:
            containers = _ContainerMetrics()
        counts = containers.counts
        lines = [
            f"Arrivi elaborati: {totals['arrivals']}",
            f"Accettate: {totals['accepted']} | Scartate: {totals['rejected']} | "
            f"Completate: {totals['completed']} | Pendenti: {totals['pending']} "
            f"(in coda: {totals['waiting']})",
            f"Cold start: {totals['cold_starts']} | Warm start: {totals['warm_starts']}",
            f"Container in creazione: {counts[ContainerState.CREATING]} | "
            f"Occupati: {counts[ContainerState.BUSY]} | Inattivi: {counts[ContainerState.IDLE]}",
            f"Container creati: {containers.created} | Rimossi: {containers.destroyed} | "
            f"Creazioni fallite: {containers.creation_failed}",
        ]
        if totals['rejections_by_reason']:
            reasons = ', '.join(f'{reason}={count}' for reason, count in sorted(self._total.rejections.items()))
            lines.append(f'Motivi di scarto: {reasons}')
        for label, mean in (
            ('Tempo medio di risposta', totals['mean_response_time_s']),
            ('Tempo medio di attesa in coda', totals['mean_queue_wait_s']),
        ):
            if mean is not None:
                lines.append(f'{label}: {mean:.6g} s')
        return '\n'.join(lines)

    @property
    def arrivals(self) -> int:
        return self._total.arrivals

    def register_node(self, node: Node, now: float) -> None:
        """Fissa lo stato iniziale delle prenotazioni del nodo."""
        if node not in self._by_node:
            self._by_node[node] = _NodeMetrics(node, now)

            for container in node.iter_containers():
                for stats in self._container_scopes(node, container.function.name):
                    stats.observe(now, container.state, 1)

    def _container_scopes(self, node: Node, name: str) -> Iterator[_ContainerMetrics]:
        self._container_functions.add(name)
        self._container_functions_by_node.setdefault(node, set()).add(name)
        for scope in ((None, None), (None, name), (node, None), (node, name)):
            if scope not in self._containers:
                self._containers[scope] = _ContainerMetrics()
            yield self._containers[scope]

    def record_container_state(
        self, node: Node, container: DockerContainer, previous: ContainerState | None, now: float,
    ) -> None:
        creation_time = None
        if previous is None:
            self._creation_started_at[container] = now
        elif previous is ContainerState.CREATING:
            started_at = self._creation_started_at.pop(container, None)
            if started_at is not None:
                creation_time = now - started_at
        for stats in self._container_scopes(node, container.function.name):
            stats.record_transition(now, previous, container.state, creation_time)
        self.observe_resources(node, now)

    def _container_snapshot(self, now: float, node: Node | None = None, name: str | None = None) -> dict:
        stats = self._containers.get((node, name))
        return (stats if stats is not None else _ContainerMetrics()).snapshot(now)

    def observe_resources(self, node: Node, now: float) -> None:
        self.register_node(node, now)
        self._by_node[node].observe(node, now)

    def _invocations_for(self, request: Request, now: float) -> Iterator[_InvocationMetrics]:
        name = request.function.name
        if name not in self._by_function:
            self._by_function[name] = _InvocationMetrics()
        yield self._total
        yield self._by_function[name]
        node = request.node
        if node is not None:
            self.observe_resources(node, now)
            stats = self._by_node[node]
            if name not in stats.by_function:
                stats.by_function[name] = _InvocationMetrics()
            yield stats.invocations
            yield stats.by_function[name]

    def record_submission(self, request: Request, now: float) -> None:
        for stats in self._invocations_for(request, now):
            stats.record_submission(request)

    def record_dequeue(self, request: Request, now: float) -> None:
        for stats in self._invocations_for(request, now):
            stats.waiting -= 1
            stats.queue_wait.add(now - request.arrival_time)
            stats.record_submission(request)

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
                node_snapshot = stats.snapshot(node, now)
                node_snapshot['containers'] = self._container_snapshot(now, node)
                for name in sorted(stats.by_function.keys() | self._container_functions_by_node.get(node, set())):
                    function_stats = node_snapshot['by_function'].setdefault(name, _InvocationMetrics().snapshot())
                    function_stats['containers'] = self._container_snapshot(now, node, name)
                nodes[node.id.key] = node_snapshot
            nodes_by_ring[ring_id] = nodes
        return {
            'time_s': now,
            'total': {**self._total.snapshot(), 'containers': self._container_snapshot(now)},
            'by_function': {
                name: {
                    **self._by_function.get(name, _InvocationMetrics()).snapshot(),
                    'containers': self._container_snapshot(now, name=name),
                }
                for name in sorted(self._by_function.keys() | self._container_functions)
            },
            'nodes_by_ring': nodes_by_ring,
        }
