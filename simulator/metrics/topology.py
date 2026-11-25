"""Collector delle metriche topologiche e di capacità allocata nel tempo (autoscaling)."""

from simulator.observer.notifications import NodeRegistered, NodeRemoved, Notification, NotificationType
from simulator.utils.welford import TimedWelford

from .base import MetricCollector, NodeKey, RingKey, timed_snapshot


class _TopologyMetrics:
    """Traccia nodi attivi, eventi di scaling e capacità (CPU·s, RAM·s) in uno scope."""

    def __init__(self) -> None:
        self.current_nodes = 0
        self.current_cpus = 0.0
        self.current_memory_mb = 0
        self.scale_out_events = 0
        self.scale_in_events = 0
        self.active_nodes = TimedWelford(0.0, 0.0)
        self.provisioned_cpus = TimedWelford(0.0, 0.0)
        self.provisioned_memory_mb = TimedWelford(0.0, 0.0)

    #################################
    ### Internal state management ###
    #################################

    def record_node_added(self, now: float, cpus: float, memory_mb: int) -> None:
        """Registra l'ingresso di un nodo (all'avvio o tramite scale-out a runtime)."""
        self.current_nodes += 1
        self.current_cpus += cpus
        self.current_memory_mb += memory_mb
        if now > 0:
            self.scale_out_events += 1
        self.active_nodes.observe(now, self.current_nodes)
        self.provisioned_cpus.observe(now, self.current_cpus)
        self.provisioned_memory_mb.observe(now, self.current_memory_mb)

    def record_node_removed(self, now: float, cpus: float, memory_mb: int) -> None:
        """Registra la rimozione di un nodo dalla topologia (scale-in)."""
        self.current_nodes = max(0, self.current_nodes - 1)
        self.current_cpus = max(0.0, self.current_cpus - cpus)
        self.current_memory_mb = max(0, self.current_memory_mb - memory_mb)
        self.scale_in_events += 1
        self.active_nodes.observe(now, self.current_nodes)
        self.provisioned_cpus.observe(now, self.current_cpus)
        self.provisioned_memory_mb.observe(now, self.current_memory_mb)

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float) -> dict:
        """Restituisce nodi attivi, eventi di scaling e integrali temporali di capacità."""
        nodes_stats = timed_snapshot(self.active_nodes, now)
        cpu_stats = timed_snapshot(self.provisioned_cpus, now)
        mem_stats = timed_snapshot(self.provisioned_memory_mb, now)

        cpu_mean = cpu_stats['mean'] or 0.0
        mem_mean = mem_stats['mean'] or 0.0
        observed = cpu_stats['observed_time_s']

        return {
            'current_active_nodes': self.current_nodes,
            'mean_active_nodes': nodes_stats['mean'],
            'scale_out_events': self.scale_out_events,
            'scale_in_events': self.scale_in_events,
            'cpu_seconds_allocated': cpu_mean * observed,
            'memory_mb_seconds_allocated': mem_mean * observed,
        }


class _TopologyMetricsCollector(MetricCollector):
    """Raccoglie metriche di topologia attiva e capacità allocata per totale, regione e ring."""

    def __init__(self) -> None:
        self._total = _TopologyMetrics()
        self._by_region: dict[str, _TopologyMetrics] = {}
        self._by_ring: dict[RingKey, _TopologyMetrics] = {}
        self._registered_nodes: set[NodeKey] = set()

    ### Identification ###

    @property
    def name(self) -> str:
        return 'topology'

    @property
    def notification_types(self) -> tuple[NotificationType, ...]:
        return (NotificationType.NODE_REGISTERED, NotificationType.NODE_REMOVED)

    ### Query state ###

    def total_snapshot(self, now: float = 0.0) -> dict:
        """Snapshot topologico complessivo della replica."""
        return self._total.to_dict(now)

    def region_snapshot(self, region: str, now: float = 0.0) -> dict:
        """Snapshot topologico per una singola regione."""
        stats = self._by_region.get(region)
        return (stats if stats is not None else _TopologyMetrics()).to_dict(now)

    def ring_snapshot(self, key: RingKey, now: float = 0.0) -> dict:
        """Snapshot topologico per un singolo anello."""
        stats = self._by_ring.get(key)
        return (stats if stats is not None else _TopologyMetrics()).to_dict(now)

    #############################
    ### Notification handling ###
    #############################

    def handle_notification(self, notification: Notification) -> None:
        """Aggiorna gli accumulatori topologici alla registrazione o rimozione di un nodo."""
        if isinstance(notification, NodeRegistered):
            region_name = notification.region.name
            ring_id = notification.ring.id
            node_key: NodeKey = (region_name, ring_id, notification.node.id.key)
            if node_key in self._registered_nodes:
                return
            self._registered_nodes.add(node_key)

            now = notification.sim_time_s
            cpus = notification.resources.total_cpus
            memory_mb = notification.resources.total_memory_mb

            self._total.record_node_added(now, cpus, memory_mb)
            self._by_region.setdefault(region_name, _TopologyMetrics()).record_node_added(now, cpus, memory_mb)
            self._by_ring.setdefault((region_name, ring_id), _TopologyMetrics()).record_node_added(now, cpus, memory_mb)
            return

        if isinstance(notification, NodeRemoved):
            region_name = notification.region.name
            ring_id = notification.ring.id
            node_key = (region_name, ring_id, notification.node.id.key)
            if node_key not in self._registered_nodes:
                return
            self._registered_nodes.remove(node_key)

            now = notification.sim_time_s
            cpus = notification.resources.total_cpus
            memory_mb = notification.resources.total_memory_mb

            self._total.record_node_removed(now, cpus, memory_mb)
            self._by_region.setdefault(region_name, _TopologyMetrics()).record_node_removed(now, cpus, memory_mb)
            self._by_ring.setdefault((region_name, ring_id), _TopologyMetrics()).record_node_removed(now, cpus, memory_mb)

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float = 0.0) -> dict:
        """Restituisce lo snapshot topologico per totale, regione e ring."""
        return {
            'total': self.total_snapshot(now),
            'by_region': {
                name: stats.to_dict(now) for name, stats in sorted(self._by_region.items())
            },
            'by_ring': {
                f'{region}/{ring_id}': stats.to_dict(now)
                for (region, ring_id), stats in sorted(self._by_ring.items())
            },
        }
