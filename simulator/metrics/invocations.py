"""Collector delle metriche relative alle invocazioni, code FIFO, offloading e latenze."""

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from simulator.observer.notifications import (
    Notification, NotificationType, RequestAccepted, RequestArrived, RequestCompleted,
    RequestDelivered, RequestDequeued, RequestForwarded, RequestQueued, RequestRejected,
    RequestRejectionReturning,
)
from simulator.utils.welford import TimedWelford, Welford

from .base import MetricCollector, NodeKey, RingKey, latency_percentiles, safe_ratio, timed_snapshot


@dataclass
class _InvocationMetrics:
    """Accumulatori online per un singolo scope di invocazioni (totale, funzione, regione, ring o nodo)."""

    in_transit: int = 0
    offloads: int = 0
    waiting: int = 0
    queued: int = 0
    dequeued: int = 0
    max_queue_length: int = 0
    queue_length: TimedWelford = field(default_factory=lambda: TimedWelford(0.0, 0.0))
    queue_wait: Welford = field(default_factory=Welford)
    queue_wait_samples: list[float] = field(default_factory=list)
    accepted: int = 0
    accepted_without_container: int = 0
    completed: int = 0
    cold_starts: int = 0
    warm_starts: int = 0
    rejections: Counter[str] = field(default_factory=Counter)
    response_time: Welford = field(default_factory=Welford)
    response_time_samples: list[float] = field(default_factory=list)
    service_time: Welford = field(default_factory=Welford)
    network_latency: Welford = field(default_factory=Welford)

    ### Statistics ###

    @property
    def arrivals(self) -> int:
        """Numero totale di invocazioni osservate nello scope."""
        return self.in_transit + self.waiting + self.accepted + self.rejections.total()

    #################################
    ### Internal state management ###
    #################################

    def record_queued(self, now: float) -> None:
        """Registra l'ingresso di una richiesta nella coda FIFO."""
        self.waiting += 1
        self.queued += 1
        if self.waiting > self.max_queue_length:
            self.max_queue_length = self.waiting
        self.queue_length.observe(now, self.waiting)

    def record_acceptance(self, now: float, warm: bool | None, queue_wait_s: float | None) -> None:
        """Registra l'ammissione di una richiesta e l'eventuale attesa in coda FIFO."""
        if queue_wait_s is not None:
            self.waiting -= 1
            self.queue_length.observe(now, self.waiting)
            self.queue_wait.add(queue_wait_s)
            self.queue_wait_samples.append(queue_wait_s)
        self.accepted += 1
        if warm is None:
            self.accepted_without_container += 1
        elif warm:
            self.warm_starts += 1
        else:
            self.cold_starts += 1

    def record_completion(
        self, response_time: float, service_time: float, network_latency_s: float = 0.0,
    ) -> None:
        """Registra il completamento con i relativi tempi di risposta, servizio e rete."""
        self.completed += 1
        self.response_time.add(response_time)
        self.response_time_samples.append(response_time)
        self.service_time.add(service_time)
        self.network_latency.add(network_latency_s)

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float = 0.0) -> dict:
        """Restituisce lo snapshot delle metriche di invocazione in formato serializzabile."""
        arrivals = self.arrivals
        rejected = self.rejections.total()
        container_starts = self.cold_starts + self.warm_starts
        queue_stats = timed_snapshot(self.queue_length, now)

        return {
            'arrivals': arrivals,
            'accepted': self.accepted,
            'rejected': rejected,
            'completed': self.completed,
            'pending': self.in_transit + self.waiting + self.accepted - self.completed,
            'in_transit': self.in_transit,
            'accepted_without_container': self.accepted_without_container,
            'waiting': self.waiting,
            'queued': self.queued,
            'dequeued': self.dequeued,
            'mean_queue_length': queue_stats['mean'],
            'max_queue_length': self.max_queue_length,
            'mean_queue_wait_s': self.queue_wait.mean,
            'queue_wait_percentiles_s': latency_percentiles(self.queue_wait_samples),
            'cold_starts': self.cold_starts,
            'warm_starts': self.warm_starts,
            'acceptance_rate': safe_ratio(self.accepted, arrivals),
            'rejection_rate': safe_ratio(rejected, arrivals),
            'cold_start_rate': safe_ratio(self.cold_starts, container_starts),
            'warm_start_rate': safe_ratio(self.warm_starts, container_starts),
            'offload_rate': safe_ratio(self.offloads, arrivals),
            'throughput_rps': safe_ratio(self.completed, now),
            'rejections_by_reason': dict(self.rejections),
            'mean_response_time_s': self.response_time.mean,
            'response_time_variance_s2': self.response_time.variance,
            'response_time_percentiles_s': latency_percentiles(self.response_time_samples),
            'mean_service_time_s': self.service_time.mean,
            'service_time_variance_s2': self.service_time.variance,
            'mean_network_latency_s': self.network_latency.mean,
        }


class _InvocationMetricsCollector(MetricCollector):
    """Raccoglie metriche di arrivo, accodamento, offloading, ammissione, rifiuto e completamento."""

    def __init__(self) -> None:
        self._total = _InvocationMetrics()
        self._by_function: dict[str, _InvocationMetrics] = {}
        self._by_region: dict[str, _InvocationMetrics] = {}
        self._by_ring: dict[RingKey, _InvocationMetrics] = {}
        self._by_node: dict[NodeKey, _InvocationMetrics] = {}
        self._by_node_function: dict[tuple[NodeKey, str], _InvocationMetrics] = {}
        self._node_functions: dict[NodeKey, set[str]] = {}
        self._external_arrivals: Counter[str] = Counter()
        self._offloads: Counter[tuple[str, str]] = Counter()
        self._intra_region_offloads: Counter[tuple[str, str, str]] = Counter()
        self._counted_intra_offloads: set[str] = set()

    ### Identification ###

    @property
    def name(self) -> str:
        return 'invocations'

    @property
    def notification_types(self) -> tuple[NotificationType, ...]:
        return (
            NotificationType.REQUEST_ARRIVED,
            NotificationType.REQUEST_FORWARDED,
            NotificationType.REQUEST_DELIVERED,
            NotificationType.REQUEST_QUEUED,
            NotificationType.REQUEST_DEQUEUED,
            NotificationType.REQUEST_ACCEPTED,
            NotificationType.REQUEST_REJECTION_RETURNING,
            NotificationType.REQUEST_REJECTED,
            NotificationType.REQUEST_COMPLETED,
        )

    ### Query state ###

    @property
    def arrivals(self) -> int:
        """Numero totale di arrivi esterni osservati nella replica."""
        return self._external_arrivals.total()

    @property
    def function_names(self) -> set[str]:
        """Nomi delle funzioni con almeno un'invocazione osservata."""
        return set(self._by_function.keys())

    def external_arrivals(self, region: str) -> int:
        """Numero di arrivi esterni nella regione indicata."""
        return self._external_arrivals[region]

    def total_snapshot(self, now: float = 0.0) -> dict:
        """Snapshot aggregato sull'intera replica."""
        return {
            **self._total.to_dict(now),
            'arrivals': self.arrivals,
            'acceptance_rate': safe_ratio(self._total.accepted, self.arrivals),
            'rejection_rate': safe_ratio(self._total.rejections.total(), self.arrivals),
            'offload_rate': safe_ratio(self._total.offloads, self.arrivals),
        }

    def function_snapshot(self, name: str, now: float = 0.0) -> dict:
        """Snapshot aggregato per una singola funzione."""
        stats = self._by_function.get(name)
        return (stats if stats is not None else _InvocationMetrics()).to_dict(now)

    def region_snapshot(self, name: str, now: float = 0.0) -> dict:
        """Snapshot aggregato per una singola regione."""
        stats = self._by_region.get(name)
        data = (stats if stats is not None else _InvocationMetrics()).to_dict(now)
        outbound_regional = sum(count for (source, _), count in self._offloads.items() if source == name)
        intra_region = sum(count for (region, _, _), count in self._intra_region_offloads.items() if region == name)
        ext = self._external_arrivals[name]
        data['offload_rate'] = safe_ratio(outbound_regional + intra_region, ext)
        return data

    def ring_snapshot(self, key: RingKey, now: float = 0.0) -> dict:
        """Snapshot aggregato per un singolo anello."""
        stats = self._by_ring.get(key)
        return (stats if stats is not None else _InvocationMetrics()).to_dict(now)

    def node_snapshot(self, key: NodeKey, now: float = 0.0) -> dict:
        """Snapshot aggregato per un singolo nodo."""
        stats = self._by_node.get(key)
        return (stats if stats is not None else _InvocationMetrics()).to_dict(now)

    def node_function_snapshot(self, key: NodeKey, name: str, now: float = 0.0) -> dict:
        """Snapshot per una singola funzione su un singolo nodo."""
        stats = self._by_node_function.get((key, name))
        return (stats if stats is not None else _InvocationMetrics()).to_dict(now)

    def node_functions(self, key: NodeKey) -> set[str]:
        """Funzioni con invocazioni osservate sul nodo indicato."""
        return set(self._node_functions.get(key, set()))

    def regional_offloads(self) -> list[dict]:
        """Elenco ordinato degli offload interregionali (origine, destinazione, conteggio)."""
        return [
            {'source': source, 'destination': destination, 'count': count}
            for (source, destination), count in sorted(self._offloads.items())
        ]

    def intra_region_offloads(self) -> list[dict]:
        """Elenco ordinato degli offload intra-regione tra nodi peer."""
        return [
            {'region': region, 'source': source, 'destination': destination, 'count': count}
            for (region, source, destination), count in sorted(self._intra_region_offloads.items())
        ]

    #############################
    ### Notification handling ###
    #############################

    def handle_notification(self, notification: Notification) -> None:
        """Aggiorna gli accumulatori in base alla notifica di richiesta ricevuta."""
        match notification:
            case RequestArrived(region=region) if region is not None:
                self._external_arrivals[region.name] += 1
                self._by_region.setdefault(region.name, _InvocationMetrics())
            case RequestForwarded(source_region=source, region=destination) if destination is not None:
                self._record_forwarded(notification.function.name, source.name, destination.name)
            case RequestDelivered():
                self._adjust_in_transit(notification, -1)
            case RequestRejectionReturning():
                self._adjust_in_transit(notification, 1)
            case RequestQueued() | RequestDequeued() | RequestAccepted() | RequestRejected() | RequestCompleted():
                if isinstance(notification, RequestRejected) and notification.request.return_delay > 0:
                    self._adjust_in_transit(notification, -1)
                self._record_outcome(notification)

    def _record_forwarded(self, function_name: str, source_name: str, destination_name: str) -> None:
        """Registra un inoltro interregionale e incrementa il contatore in transito."""
        self._offloads[source_name, destination_name] += 1
        for stats in self._scopes(function_name, destination_name):
            stats.in_transit += 1
            stats.offloads += 1

    def _adjust_in_transit(
        self,
        notification: RequestDelivered | RequestRejectionReturning | RequestRejected,
        delta: int,
    ) -> None:
        """Aggiorna il numero di richieste in transito negli scope della notifica."""
        region_name = notification.region.name if notification.region is not None else None
        for stats in self._scopes(notification.function.name, region_name):
            stats.in_transit += delta

    def _track_intra_region_offload(
        self, event: RequestQueued | RequestAccepted | RequestRejected,
    ) -> None:
        """Registra una sola volta gli offload verso un nodo peer nella stessa regione."""
        request = event.request
        source = request.offloaded_from
        dest = event.node
        if (
            source is None
            or dest is None
            or source is dest
            or event.region is None
            or source.region is not event.region
            or request.id in self._counted_intra_offloads
        ):
            return
        self._counted_intra_offloads.add(request.id)
        self._intra_region_offloads[event.region.name, source.id.key, dest.id.key] += 1
        for stats in self._scopes(event.function.name, event.region.name):
            stats.offloads += 1

    def _record_outcome(
        self,
        event: RequestQueued | RequestDequeued | RequestAccepted | RequestRejected | RequestCompleted,
    ) -> None:
        """Aggiorna tutti gli scope interessati dall'esito della richiesta."""
        if isinstance(event, (RequestQueued, RequestAccepted, RequestRejected)):
            self._track_intra_region_offload(event)

        region_name = event.region.name if event.region is not None else None
        ring_id = event.ring.id if event.ring is not None else None
        node_id = event.node.id.key if event.node is not None else None
        now = event.sim_time_s

        for stats in self._scopes(event.function.name, region_name, ring_id, node_id):
            match event:
                case RequestQueued():
                    stats.record_queued(now)
                case RequestDequeued():
                    stats.dequeued += 1
                case RequestAccepted():
                    stats.record_acceptance(now, event.warm, event.queue_wait_s)
                case RequestRejected():
                    stats.rejections[event.reason] += 1
                case RequestCompleted():
                    if event.service_time_s is None:
                        raise ValueError('Il completamento richiede una durata di servizio')
                    network_latency = event.request.return_delay * 2
                    stats.record_completion(event.response_time_s, event.service_time_s, network_latency)

    def _scopes(
        self,
        function_name: str,
        region: str | None,
        ring_id: str | None = None,
        node_id: str | None = None,
    ) -> Iterator[_InvocationMetrics]:
        """Restituisce tutti gli accumulatori gerarchici da aggiornare per l'evento."""
        if function_name not in self._by_function:
            self._by_function[function_name] = _InvocationMetrics()
        yield self._total
        yield self._by_function[function_name]

        if region is not None:
            if region not in self._by_region:
                self._by_region[region] = _InvocationMetrics()
            yield self._by_region[region]

        if region is not None and ring_id is not None:
            ring_key: RingKey = (region, ring_id)
            if ring_key not in self._by_ring:
                self._by_ring[ring_key] = _InvocationMetrics()
            yield self._by_ring[ring_key]

        if region is not None and ring_id is not None and node_id is not None:
            key: NodeKey = (region, ring_id, node_id)
            if key not in self._by_node:
                self._by_node[key] = _InvocationMetrics()
            self._node_functions.setdefault(key, set()).add(function_name)
            node_fn_key = (key, function_name)
            if node_fn_key not in self._by_node_function:
                self._by_node_function[node_fn_key] = _InvocationMetrics()
            yield self._by_node[key]
            yield self._by_node_function[node_fn_key]

    #####################
    ### Serialization ###
    #####################

    def to_dict(self, now: float = 0.0) -> dict:
        """Restituisce tutte le metriche di invocazione raccolte dal collector."""
        return {
            'total': self.total_snapshot(now),
            'by_function': {
                name: stats.to_dict(now) for name, stats in sorted(self._by_function.items())
            },
            'by_region': {
                name: {
                    'external_arrivals': self._external_arrivals[name],
                    **self.region_snapshot(name, now),
                }
                for name in sorted(self._by_region)
            },
            'by_ring': {
                f'{region}/{ring_id}': stats.to_dict(now)
                for (region, ring_id), stats in sorted(self._by_ring.items())
            },
            'regional_offloads': self.regional_offloads(),
            'intra_region_offloads': self.intra_region_offloads(),
        }
