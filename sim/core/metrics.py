"""Metriche delle invocazioni e delle prenotazioni Serverledge."""

from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field

from sim.model.container import ContainerState
from sim.model.infrastructure import Infrastructure
from sim.model.region import Region
from sim.observer.notifications import (
    ContainerStateChanged, NodeRegistered, Notification, RequestAccepted, RequestArrived, RequestCompleted,
    RequestDelivered, RequestForwarded, RequestQueued, RequestRejected,
)
from sim.observer.notifications.node import ResourceSnapshot
from sim.utils.welford import TimedWelford, Welford


@dataclass
class _InvocationMetrics:
    in_transit: int = 0
    waiting: int = 0
    queued: int = 0
    queue_wait: Welford = field(default_factory=Welford)
    accepted: int = 0
    accepted_without_container: int = 0
    completed: int = 0
    cold_starts: int = 0
    warm_starts: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    response_time: Welford = field(default_factory=Welford)
    service_time: Welford = field(default_factory=Welford)

    @property
    def arrivals(self) -> int:
        return self.in_transit + self.waiting + self.accepted + self.rejections.total()

    def record_acceptance(self, warm: bool | None, queue_wait_s: float | None) -> None:
        if queue_wait_s is not None:
            self.waiting -= 1
            self.queue_wait.add(queue_wait_s)
        self.accepted += 1
        if warm is None:
            self.accepted_without_container += 1
        elif warm:
            self.warm_starts += 1
        else:
            self.cold_starts += 1

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
            'pending': self.in_transit + self.waiting + self.accepted - self.completed,
            'in_transit': self.in_transit,
            'accepted_without_container': self.accepted_without_container,
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
    def __init__(self, resources: ResourceSnapshot, now: float) -> None:
        self.invocations = _InvocationMetrics()
        self.by_function: dict[str, _InvocationMetrics] = {}
        self.current = resources
        self.resources = {
            name: TimedWelford(now, value) for name, value in self._reservations(resources).items()
        }

    @staticmethod
    def _reservations(resources: ResourceSnapshot) -> dict[str, float]:
        return {
            'used_cpus': resources.used_cpus,
            'busy_memory_mb': resources.busy_memory_mb,
            'warm_memory_mb': resources.warm_memory_mb,
        }

    def observe(self, resources: ResourceSnapshot, now: float) -> None:
        self.current = resources
        for name, value in self._reservations(resources).items():
            self.resources[name].observe(now, value)

    def snapshot(self, now: float) -> dict:
        return {
            **self._reservations(self.current),
            'total_cpus': self.current.total_cpus,
            'total_memory_mb': self.current.total_memory_mb,
            'invocations': self.invocations.snapshot(),
            'by_function': {name: stats.snapshot() for name, stats in self.by_function.items()},
            'time_weighted_resources': {
                name: _timed_snapshot(accumulator, now) for name, accumulator in self.resources.items()
            },
        }


type _NodeKey = tuple[str, str, str]  # Regione, ring e nome del nodo.


class SimulationMetrics:
    """Aggrega le notifiche della replica in statistiche online."""

    def __init__(self) -> None:
        self._total = _InvocationMetrics()
        self._by_function: dict[str, _InvocationMetrics] = {}
        self._by_region: dict[str, _InvocationMetrics] = {}
        self._external_arrivals: Counter[str] = Counter()
        self._offloads: Counter[tuple[str, str]] = Counter()
        self._by_node: dict[_NodeKey, _NodeMetrics] = {}
        self._containers: dict[tuple[_NodeKey | None, str | None], _ContainerMetrics] = {}
        self._container_functions_by_node: dict[_NodeKey, set[str]] = {}
        self._container_functions: set[str] = set()
        self._creation_started_at: dict[tuple[_NodeKey, str], float] = {}

    @property
    def arrivals(self) -> int:
        return self._external_arrivals.total()

    def on_notification(self, notification: Notification) -> None:
        """Aggiorna le metriche prima che la notifica raggiunga i subscriber esterni."""
        match notification:
            case NodeRegistered():
                self._register_node(notification)
            case ContainerStateChanged():
                self._record_container_state(notification)
            case RequestArrived(region=region) if region is not None:
                self._external_arrivals[region] += 1
            case RequestForwarded(source_region=source, region=destination) if destination is not None:
                self._offloads[source, destination] += 1
                for stats in self._invocations_for(notification.function, destination):
                    stats.in_transit += 1
            case RequestDelivered():
                for stats in self._invocations_for(notification.function, notification.region):
                    stats.in_transit -= 1
            case RequestQueued() | RequestAccepted() | RequestRejected() | RequestCompleted():
                self._record_invocation(notification)

    def _record_invocation(
        self, event: RequestQueued | RequestAccepted | RequestRejected | RequestCompleted,
    ) -> None:
        for stats in self._invocations_for(event.function, event.region, event.ring_id, event.node_id):
            match event:
                case RequestQueued():
                    stats.waiting += 1
                    stats.queued += 1
                case RequestAccepted():
                    stats.record_acceptance(event.warm, event.queue_wait_s)
                case RequestRejected():
                    stats.rejections[event.reason] += 1
                case RequestCompleted():
                    if event.service_time_s is None:
                        raise ValueError('Il completamento richiede una durata di servizio')
                    stats.record_completion(event.response_time_s, event.service_time_s)

    def _register_node(self, event: NodeRegistered) -> None:
        key = (event.region, event.ring_id, event.node_id)
        if key in self._by_node:
            return
        self._by_node[key] = _NodeMetrics(event.resources, event.sim_time_s)
        for container in event.containers:
            for stats in self._container_scopes(key, container.function):
                stats.observe(event.sim_time_s, container.state, 1)

    def _container_scopes(self, key: _NodeKey, name: str) -> Iterator[_ContainerMetrics]:
        self._container_functions.add(name)
        self._container_functions_by_node.setdefault(key, set()).add(name)
        for scope in ((None, None), (None, name), (key, None), (key, name)):
            if scope not in self._containers:
                self._containers[scope] = _ContainerMetrics()
            yield self._containers[scope]

    def _record_container_state(self, event: ContainerStateChanged) -> None:
        if event.region is None or event.state is None or event.resources is None:
            raise ValueError('La transizione richiede regione, stato e prenotazioni del nodo')
        key = (event.region, event.ring_id, event.node_id)
        container = (key, event.container_id)
        now = event.sim_time_s
        creation_time = None
        if event.previous is None:
            self._creation_started_at[container] = now
        elif event.previous is ContainerState.CREATING:
            started_at = self._creation_started_at.pop(container, None)
            if started_at is not None:
                creation_time = now - started_at
        for stats in self._container_scopes(key, event.function):
            stats.record_transition(now, event.previous, event.state, creation_time)
        self._by_node[key].observe(event.resources, now)

    def _container_snapshot(self, now: float, key: _NodeKey | None = None, name: str | None = None) -> dict:
        stats = self._containers.get((key, name))
        return (stats if stats is not None else _ContainerMetrics()).snapshot(now)

    def _invocations_for(
        self, name: str, region: str | None, ring_id: str | None = None, node_id: str | None = None,
    ) -> Iterator[_InvocationMetrics]:
        if name not in self._by_function:
            self._by_function[name] = _InvocationMetrics()
        yield self._total
        yield self._by_function[name]
        if region is not None:
            if region not in self._by_region:
                self._by_region[region] = _InvocationMetrics()
            yield self._by_region[region]
        if region is not None and ring_id is not None and node_id is not None:
            stats = self._by_node[region, ring_id, node_id]
            if name not in stats.by_function:
                stats.by_function[name] = _InvocationMetrics()
            yield stats.invocations
            yield stats.by_function[name]

    def _nodes_snapshot(self, now: float, region: str, infrastructure: Infrastructure | None) -> dict:
        """Restituisce e separa le metriche dei nodi per ring e nome."""
        nodes_by_ring = {}
        if infrastructure is None:
            return nodes_by_ring
        for ring_id, ring in infrastructure.rings.items():
            nodes = {}
            for node in ring:
                key = (region, ring_id, node.id.key)
                stats = self._by_node.get(key)
                if stats is None:
                    stats = _NodeMetrics(ResourceSnapshot.from_node(node), now)
                node_snapshot = stats.snapshot(now)
                node_snapshot['containers'] = self._container_snapshot(now, key)
                for name in sorted(stats.by_function.keys() | self._container_functions_by_node.get(key, set())):
                    function_stats = node_snapshot['by_function'].setdefault(name, _InvocationMetrics().snapshot())
                    function_stats['containers'] = self._container_snapshot(now, key, name)
                nodes[node.id.key] = node_snapshot
            nodes_by_ring[ring_id] = nodes
        return nodes_by_ring

    def snapshot(self, now: float, regions: Mapping[str, Region], primary_region: str) -> dict:
        regional = {
            name: {
                'mode': region.mode.value,
                'external_arrivals': self._external_arrivals[name],
                'invocations': (
                    self._by_region[name].snapshot() if name in self._by_region else _InvocationMetrics().snapshot()
                ),
                'nodes_by_ring': self._nodes_snapshot(now, name, region.infrastructure),
            }
            for name, region in regions.items()
        }
        return {
            'time_s': now,
            'total': {**self._total.snapshot(), 'arrivals': self.arrivals, 'containers': self._container_snapshot(now)},
            'by_function': {
                name: {
                    **self._by_function.get(name, _InvocationMetrics()).snapshot(),
                    'containers': self._container_snapshot(now, name=name),
                }
                for name in sorted(self._by_function.keys() | self._container_functions)
            },
            'primary_region': primary_region,
            'nodes_by_ring': regional[primary_region]['nodes_by_ring'],
            'regions': regional,
            'regional_offloads': [
                {'source': source, 'destination': destination, 'count': count}
                for (source, destination), count in sorted(self._offloads.items())
            ],
        }
