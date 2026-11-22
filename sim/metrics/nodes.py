"""Collector delle metriche di prenotazione e utilizzo delle risorse dei nodi."""

from sim.model.node import Node
from sim.observer.notifications import (
    ContainerStateChanged, NodeRegistered, Notification, NotificationType,
)
from sim.observer.notifications.node import ResourceSnapshot
from sim.utils.welford import TimedWelford
from .base import MetricCollector, NodeKey, safe_ratio, timed_snapshot


class _NodeResourceMetrics:
    """Traccia lo stato corrente e la media temporale delle risorse prenotate su un nodo."""

    def __init__(self, resources: ResourceSnapshot, now: float) -> None:
        self.current = resources
        self.resources = {
            name: TimedWelford(now, value)
            for name, value in self._reservations(resources).items()
        }

    @staticmethod
    def _reservations(resources: ResourceSnapshot) -> dict[str, float]:
        """Estrae i contatori di prenotazione monitorati nel tempo."""
        return {
            'used_cpus': resources.used_cpus,
            'busy_memory_mb': resources.busy_memory_mb,
            'warm_memory_mb': resources.warm_memory_mb,
        }

    #################################
    ### Internal state management ###
    #################################

    def observe(self, resources: ResourceSnapshot, now: float) -> None:
        """Aggiorna le prenotazioni correnti e gli accumulatori pesati nel tempo."""
        self.current = resources
        for name, value in self._reservations(resources).items():
            self.resources[name].observe(now, value)

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float) -> dict:
        """Restituisce prenotazioni correnti, capacità totali, medie temporali e utilizzi."""
        time_weighted = {
            name: timed_snapshot(accumulator, now)
            for name, accumulator in self.resources.items()
        }
        mean_cpus = time_weighted['used_cpus']['mean']
        mean_busy_mem = time_weighted['busy_memory_mb']['mean']
        mean_warm_mem = time_weighted['warm_memory_mb']['mean']

        return {
            **self._reservations(self.current),
            'total_cpus': self.current.total_cpus,
            'total_memory_mb': self.current.total_memory_mb,
            'mean_cpu_utilization': (
                safe_ratio(mean_cpus, self.current.total_cpus) if mean_cpus is not None else None
            ),
            'mean_busy_memory_utilization': (
                safe_ratio(mean_busy_mem, self.current.total_memory_mb) if mean_busy_mem is not None else None
            ),
            'mean_warm_memory_utilization': (
                safe_ratio(mean_warm_mem, self.current.total_memory_mb) if mean_warm_mem is not None else None
            ),
            'time_weighted_resources': time_weighted,
        }


class _NodeMetricsCollector(MetricCollector):
    """Raccoglie le prenotazioni di CPU e memoria per ogni nodo registrato nella replica."""

    # FIXME (scale-in/scale-out): gestire NodeRemoved per chiudere TimedWelford al momento della
    # rimozione del nodo e distinguere istanze che riusano lo stesso nome durante il drain.

    def __init__(self) -> None:
        self._by_node: dict[NodeKey, _NodeResourceMetrics] = {}

    ### Identification ###

    @property
    def name(self) -> str:
        return 'nodes'

    @property
    def notification_types(self) -> tuple[NotificationType, ...]:
        return (
            NotificationType.NODE_REGISTERED,
            NotificationType.CONTAINER_STATE_CHANGED,
            NotificationType.CONTAINER_STARTED,
            NotificationType.CONTAINER_STOPPED,
        )

    ### Query state ###

    def resource_snapshot(self, key: NodeKey, node: Node, now: float) -> dict:
        """Restituisce lo snapshot delle risorse del nodo indicato, inizializzandolo se assente."""
        stats = self._by_node.get(key)
        if stats is None:
            stats = _NodeResourceMetrics(ResourceSnapshot.from_node(node), now)
        return stats.to_dict(now)

    #############################
    ### Notification handling ###
    #############################

    def handle_notification(self, notification: Notification) -> None:
        """Aggiorna lo stato delle risorse del nodo in seguito a registrazione o transizione container."""
        match notification:
            case NodeRegistered():
                key: NodeKey = (notification.region.name, notification.ring.id, notification.node.id.key)
                if key not in self._by_node:
                    self._by_node[key] = _NodeResourceMetrics(notification.resources, notification.sim_time_s)
            case ContainerStateChanged():
                if notification.region is None or notification.resources is None:
                    raise ValueError('La transizione richiede regione e prenotazioni del nodo')
                key = (notification.region.name, notification.ring.id, notification.node.id.key)
                if key not in self._by_node:
                    self._by_node[key] = _NodeResourceMetrics(notification.resources, notification.sim_time_s)
                else:
                    self._by_node[key].observe(notification.resources, notification.sim_time_s)

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float = 0.0) -> dict:
        """Restituisce le metriche di risorse per tutti i nodi osservati."""
        return {
            f'{region}/{ring_id}/{node_id}': stats.to_dict(now)
            for (region, ring_id, node_id), stats in sorted(self._by_node.items())
        }
