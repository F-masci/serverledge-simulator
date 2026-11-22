"""Coordinatore centrale delle metriche online alimentate dall'observer."""

from collections.abc import Mapping
import json
from types import MappingProxyType

from sim.model.infrastructure import Infrastructure
from sim.model.region import Region
from sim.observer.notifications import Notification, NotificationType
from sim.utils.report import format_metrics
from .base import MetricCollector, NodeKey, RingKey, safe_ratio
from .containers import _ContainerMetricsCollector
from .invocations import _InvocationMetricsCollector
from .nodes import _NodeMetricsCollector
from .topology import _TopologyMetricsCollector


class SimulationMetrics:
    """
    Dispatcher centrale delle metriche online della replica.

    Riceve le notifiche dall'observer e le distribuisce ai collector privati
    specializzati (invocazioni, container, nodi, topologia) in base al NotificationType.
    Espone metodi per ispezionare, stampare e serializzare sia i singoli
    collector sia lo snapshot complessivo della simulazione.
    """

    def __init__(self) -> None:
        # Collector specializzati per categoria di metrica.
        self._invocations = _InvocationMetricsCollector()
        self._containers = _ContainerMetricsCollector()
        self._nodes = _NodeMetricsCollector()
        self._topology = _TopologyMetricsCollector()

        # Registro indicizzato per nome e tabella di routing per tipo di notifica.
        collectors: tuple[MetricCollector, ...] = (
            self._invocations, self._containers, self._nodes, self._topology,
        )
        self._collectors: dict[str, MetricCollector] = {
            collector.name: collector for collector in collectors
        }
        self._dispatch: dict[NotificationType, tuple[MetricCollector, ...]] = self._build_dispatch(collectors)

    @staticmethod
    def _build_dispatch(
        collectors: tuple[MetricCollector, ...],
    ) -> dict[NotificationType, tuple[MetricCollector, ...]]:
        """Costruisce la mappa NotificationType -> collector interessati."""
        routing: dict[NotificationType, list[MetricCollector]] = {}
        for collector in collectors:
            for notification_type in collector.notification_types:
                routing.setdefault(notification_type, []).append(collector)
        return {
            notification_type: tuple(targets)
            for notification_type, targets in routing.items()
        }

    ### Metrics State ###

    @property
    def arrivals(self) -> int:
        """Numero totale di arrivi esterni osservati nella replica."""
        return self._invocations.arrivals

    @property
    def collectors(self) -> Mapping[str, MetricCollector]:
        """Vista non modificabile dei collector registrati per nome."""
        return MappingProxyType(self._collectors)

    #############################
    ### Notification handling ###
    #############################

    def on_notification(self, notification: Notification) -> None:
        """Distribuisce la notifica ai collector registrati per il suo tipo."""
        for collector in self._dispatch.get(notification.type, ()):
            collector.handle_notification(notification)

    #####################
    ### Serialization ###
    #####################

    def collector_dict(self, name: str, now: float = 0.0) -> dict:
        """Restituisce lo snapshot serializzabile di un singolo collector."""
        if name not in self._collectors:
            raise KeyError(f'Collector di metriche sconosciuto: {name}')
        return self._collectors[name].to_dict(now)

    def format_collector(self, name: str, now: float = 0.0) -> str:
        """Restituisce la rappresentazione testuale stampabile di un singolo collector."""
        if name not in self._collectors:
            raise KeyError(f'Collector di metriche sconosciuto: {name}')
        return self._collectors[name].format_summary(now)

    @staticmethod
    def _aggregate_resources(node_snapshots: list[dict]) -> dict:
        """Aggrega capacità e utilizzo medio delle risorse su un insieme di nodi."""
        total_cpus = sum(node['total_cpus'] for node in node_snapshots)
        total_memory_mb = sum(node['total_memory_mb'] for node in node_snapshots)
        mean_cpus = [
            node['time_weighted_resources']['used_cpus']['mean']
            for node in node_snapshots
            if node['time_weighted_resources']['used_cpus']['mean'] is not None
        ]
        mean_busy_mem = [
            node['time_weighted_resources']['busy_memory_mb']['mean']
            for node in node_snapshots
            if node['time_weighted_resources']['busy_memory_mb']['mean'] is not None
        ]
        mean_warm_mem = [
            node['time_weighted_resources']['warm_memory_mb']['mean']
            for node in node_snapshots
            if node['time_weighted_resources']['warm_memory_mb']['mean'] is not None
        ]
        return {
            'total_cpus': total_cpus,
            'total_memory_mb': total_memory_mb,
            'mean_cpu_utilization': safe_ratio(sum(mean_cpus), total_cpus) if mean_cpus else None,
            'mean_busy_memory_utilization': safe_ratio(sum(mean_busy_mem), total_memory_mb) if mean_busy_mem else None,
            'mean_warm_memory_utilization': safe_ratio(sum(mean_warm_mem), total_memory_mb) if mean_warm_mem else None,
        }

    @staticmethod
    def _load_balance_ratio(node_snapshots: list[dict]) -> float | None:
        """Rapporto max / media delle invocazioni accettate sui nodi del ring (1.0 = bilanciamento perfetto)."""
        if not node_snapshots:
            return None
        counts = [node['invocations']['accepted'] for node in node_snapshots]
        total = sum(counts)
        if total == 0:
            return None
        mean = total / len(counts)
        return max(counts) / mean

    def _nodes_snapshot(self, now: float, region: str, infrastructure: Infrastructure | None) -> dict:
        """Compone le metriche di risorse, invocazioni e container per ring e nodo."""
        nodes_by_ring: dict[str, dict] = {}
        if infrastructure is None:
            return nodes_by_ring

        for ring_id, ring in infrastructure.rings.items():
            nodes: dict[str, dict] = {}
            for node in ring:
                key: NodeKey = (region, ring_id, node.id.key)
                node_functions = sorted(
                    self._invocations.node_functions(key) | self._containers.node_functions(key)
                )
                node_snapshot = {
                    **self._nodes.resource_snapshot(key, node, now),
                    'invocations': self._invocations.node_snapshot(key, now),
                    'by_function': {
                        name: {
                            **self._invocations.node_function_snapshot(key, name, now),
                            'containers': self._containers.snapshot_for(key, name, now),
                        }
                        for name in node_functions
                    },
                    'containers': self._containers.snapshot_for(key, now=now),
                }
                nodes[node.id.key] = node_snapshot
            nodes_by_ring[ring_id] = nodes
        return nodes_by_ring

    def _rings_snapshot(self, now: float, region: str, nodes_by_ring: dict[str, dict]) -> dict:
        """Compone le metriche aggregate per ciascun ring della regione."""
        rings: dict[str, dict] = {}
        for ring_id, nodes in nodes_by_ring.items():
            ring_key: RingKey = (region, ring_id)
            node_list = list(nodes.values())
            rings[ring_id] = {
                'nodes_count': len(node_list),
                'invocations': self._invocations.ring_snapshot(ring_key, now),
                'containers': self._containers.ring_snapshot(ring_key, now),
                'resources': self._aggregate_resources(node_list),
                'load_balance_ratio': self._load_balance_ratio(node_list),
                'topology': self._topology.ring_snapshot(ring_key, now),
            }
        return rings

    def snapshot(self, now: float, regions: Mapping[str, Region], primary_region: str) -> dict:
        """Compone lo snapshot complessivo della replica per il resoconto finale."""
        regional: dict[str, dict] = {}
        for name, region in regions.items():
            nodes_by_ring = self._nodes_snapshot(now, name, region.infrastructure)
            all_region_nodes = [node for nodes in nodes_by_ring.values() for node in nodes.values()]
            regional[name] = {
                'mode': region.mode.value,
                'external_arrivals': self._invocations.external_arrivals(name),
                'invocations': self._invocations.region_snapshot(name, now),
                'resources': self._aggregate_resources(all_region_nodes),
                'topology': self._topology.region_snapshot(name, now),
                'rings': self._rings_snapshot(now, name, nodes_by_ring),
                'nodes_by_ring': nodes_by_ring,
            }

        all_functions = sorted(self._invocations.function_names | self._containers.function_names)
        return {
            'time_s': now,
            'total': {
                **self._invocations.total_snapshot(now),
                'containers': self._containers.snapshot_for(now=now),
            },
            'by_function': {
                name: {
                    **self._invocations.function_snapshot(name, now),
                    'containers': self._containers.snapshot_for(name=name, now=now),
                }
                for name in all_functions
            },
            'primary_region': primary_region,
            'nodes_by_ring': regional[primary_region]['nodes_by_ring'],
            'regions': regional,
            'regional_offloads': self._invocations.regional_offloads(),
            'intra_region_offloads': self._invocations.intra_region_offloads(),
            'topology': self._topology.total_snapshot(now),
        }

    def format_summary(self, now: float, regions: Mapping[str, Region], primary_region: str) -> str:
        """Restituisce il report testuale completo della replica."""
        return format_metrics(self.snapshot(now, regions, primary_region))

    def to_dict(self, now: float = 0.0) -> dict:
        """Restituisce lo snapshot aggregato di tutti i singoli collector."""
        return {
            name: collector.to_dict(now)
            for name, collector in self._collectors.items()
        }

    def to_json(self, now: float = 0.0) -> str:
        """Serializza in JSON lo snapshot aggregato di tutti i collector."""
        return json.dumps(self.to_dict(now), ensure_ascii=False, allow_nan=False)
