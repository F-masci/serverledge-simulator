"""Collector delle metriche del ciclo di vita, riuso e permanenza idle dei container."""

from collections.abc import Iterator
import json

from sim.model.container import ContainerState
from sim.observer.notifications import (
    ContainerStateChanged, NodeRegistered, Notification, NotificationType,
)
from sim.utils.welford import TimedWelford, Welford
from .base import MetricCollector, NodeKey, RingKey, safe_ratio, timed_snapshot

# Stati attivi monitorati tramite conteggio istantaneo e media temporale.
_LIVE_STATES = (ContainerState.CREATING, ContainerState.BUSY, ContainerState.IDLE)


class _ContainerMetrics:
    """Accumulatori online del ciclo di vita dei container per uno specifico scope."""

    def __init__(self) -> None:
        self.counts = dict.fromkeys(_LIVE_STATES, 0)
        self.time_weighted = {state: TimedWelford(0, 0) for state in _LIVE_STATES}
        self.creations_started = 0
        self.created = 0
        self.creation_failed = 0
        self.destroyed = 0
        self.expired = 0
        self.evicted = 0
        self.activations = 0
        self.creation_time = Welford()
        self.idle_duration = Welford()

    #################################
    ### Internal state management ###
    #################################

    def observe(self, now: float, state: ContainerState, delta: int) -> None:
        """Aggiorna il conteggio dello stato indicato e la sua media pesata nel tempo."""
        self.counts[state] += delta
        self.time_weighted[state].observe(now, self.counts[state])

    def record_transition(
        self,
        now: float,
        previous: ContainerState | None,
        state: ContainerState,
        creation_time: float | None,
        idle_duration: float | None = None,
        evicted: bool = False,
    ) -> None:
        """Registra una transizione di stato del container e i relativi contatori."""
        if previous is not None:
            self.observe(now, previous, -1)
        if state in self.counts:
            self.observe(now, state, 1)

        if idle_duration is not None:
            self.idle_duration.add(idle_duration)

        if state is ContainerState.BUSY and previous in (ContainerState.CREATING, ContainerState.IDLE):
            self.activations += 1

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
            if evicted:
                self.evicted += 1
            else:
                self.expired += 1

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float) -> dict:
        """Restituisce lo snapshot delle metriche dei container all'istante now."""
        return {
            **{state.value: count for state, count in self.counts.items()},
            'creations_started': self.creations_started,
            'created': self.created,
            'creation_failed': self.creation_failed,
            'destroyed': self.destroyed,
            'expired': self.expired,
            'evicted': self.evicted,
            'eviction_rate': safe_ratio(self.evicted, self.created),
            'expiration_rate': safe_ratio(self.expired, self.created),
            'mean_activations_per_container': safe_ratio(self.activations, self.created),
            'mean_creation_time_s': self.creation_time.mean,
            'creation_time_variance_s2': self.creation_time.variance,
            'mean_idle_duration_s': self.idle_duration.mean,
            'idle_duration_variance_s2': self.idle_duration.variance,
            'time_weighted_counts': {
                state.value: timed_snapshot(accumulator, now)
                for state, accumulator in self.time_weighted.items()
            },
        }

    def to_json(self, now: float) -> str:
        """Serializza in JSON lo snapshot delle metriche dei container."""
        return json.dumps(self.to_dict(now), ensure_ascii=False, allow_nan=False)


class _ContainerMetricsCollector(MetricCollector):
    """Raccoglie conteggi, transizioni, eviction, tempi di creazione e permanenza idle dei container."""

    def __init__(self) -> None:
        self._containers: dict[tuple[NodeKey | None, str | None], _ContainerMetrics] = {}
        self._by_ring: dict[RingKey, _ContainerMetrics] = {}
        self._container_functions: set[str] = set()
        self._container_functions_by_node: dict[NodeKey, set[str]] = {}
        self._creation_started_at: dict[tuple[NodeKey, str], float] = {}
        self._idle_started_at: dict[tuple[NodeKey, str], float] = {}
        self._registered_nodes: set[NodeKey] = set()

    ### Identification ###

    @property
    def name(self) -> str:
        return 'containers'

    @property
    def notification_types(self) -> tuple[NotificationType, ...]:
        return (
            NotificationType.NODE_REGISTERED,
            NotificationType.CONTAINER_STATE_CHANGED,
            NotificationType.CONTAINER_STARTED,
            NotificationType.CONTAINER_STOPPED,
        )

    ### Query state ###

    @property
    def function_names(self) -> set[str]:
        """Funzioni con almeno un container osservato."""
        return set(self._container_functions)

    def node_functions(self, key: NodeKey) -> set[str]:
        """Funzioni con almeno un container osservato sul nodo indicato."""
        return set(self._container_functions_by_node.get(key, set()))

    def snapshot_for(
        self, now: float, key: NodeKey | None = None, name: str | None = None,
    ) -> dict:
        """Restituisce lo snapshot container per lo scope (nodo, funzione) richiesto."""
        stats = self._containers.get((key, name))
        return (stats if stats is not None else _ContainerMetrics()).to_dict(now)

    def ring_snapshot(self, now: float, key: RingKey) -> dict:
        """Restituisce lo snapshot container aggregato per il ring indicato."""
        stats = self._by_ring.get(key)
        return (stats if stats is not None else _ContainerMetrics()).to_dict(now)

    #############################
    ### Notification handling ###
    #############################

    def handle_notification(self, notification: Notification) -> None:
        """Elabora registrazioni iniziali dei nodi e transizioni di stato dei container."""
        match notification:
            case NodeRegistered():
                self._register_node_containers(notification)
            case ContainerStateChanged():
                self._record_transition(notification)

    def _register_node_containers(self, event: NodeRegistered) -> None:
        """Registra eventuali container già presenti sul nodo al momento dell'iscrizione."""
        key: NodeKey = (event.region.name, event.ring.id, event.node.id.key)
        if key in self._registered_nodes:
            return
        self._registered_nodes.add(key)

        for snapshot in event.containers:
            if snapshot.state is ContainerState.IDLE:
                self._idle_started_at[key, snapshot.container.id] = event.sim_time_s
            for stats in self._scopes(key, snapshot.container.function.name):
                stats.observe(event.sim_time_s, snapshot.state, 1)

    def _record_transition(self, event: ContainerStateChanged) -> None:
        """Aggiorna gli accumulatori di tutti gli scope per la transizione del container."""
        if event.region is None or event.state is None:
            raise ValueError('La transizione richiede regione e stato del container')

        key: NodeKey = (event.region.name, event.ring.id, event.node.id.key)
        container_key = (key, event.container.id)
        now = event.sim_time_s
        creation_time = None
        idle_duration = None
        evicted = False

        # Traccia la durata del cold start tra prenotazione (None) e avvio.
        if event.previous is None:
            self._creation_started_at[container_key] = now
        elif event.previous is ContainerState.CREATING:
            started_at = self._creation_started_at.pop(container_key, None)
            if started_at is not None:
                creation_time = now - started_at

        # Traccia la permanenza in stato IDLE prima di riattivazione o distruzione.
        if event.state is ContainerState.IDLE:
            self._idle_started_at[container_key] = now
        elif event.previous is ContainerState.IDLE:
            idle_since = self._idle_started_at.pop(container_key, None)
            if idle_since is not None:
                idle_duration = now - idle_since
            # Distingue eviction forzata (prima della scadenza) da scadenza ordinaria per idle_timeout.
            if event.state is ContainerState.DESTROYED:
                expiration = event.container.expiration_time
                evicted = expiration is None or now <= expiration

        for stats in self._scopes(key, event.function.name):
            stats.record_transition(
                now, event.previous, event.state, creation_time,
                idle_duration=idle_duration, evicted=evicted,
            )

    def _scopes(self, key: NodeKey, function_name: str) -> Iterator[_ContainerMetrics]:
        """Restituisce gli scope gerarchici: totale, funzione, ring, nodo, nodo+funzione."""
        self._container_functions.add(function_name)
        self._container_functions_by_node.setdefault(key, set()).add(function_name)
        for scope in ((None, None), (None, function_name), (key, None), (key, function_name)):
            if scope not in self._containers:
                self._containers[scope] = _ContainerMetrics()
            yield self._containers[scope]

        ring_key: RingKey = (key[0], key[1])
        if ring_key not in self._by_ring:
            self._by_ring[ring_key] = _ContainerMetrics()
        yield self._by_ring[ring_key]

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float = 0.0) -> dict:
        """Restituisce lo snapshot aggregato dei container per totale, funzione e ring."""
        return {
            'total': self.snapshot_for(now),
            'by_function': {
                name: self.snapshot_for(now, name=name)
                for name in sorted(self._container_functions)
            },
            'by_ring': {
                f'{region}/{ring_id}': stats.to_dict(now)
                for (region, ring_id), stats in sorted(self._by_ring.items())
            },
        }
