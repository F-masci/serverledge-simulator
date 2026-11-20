"""Coordinamento degli arrivi sul calendario della simulazione."""

from copy import deepcopy
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from math import floor
from types import MappingProxyType
from uuid import uuid4

from sim.io.scenario import Scenario
from sim.model.container import ContainerState, DockerContainer
from sim.model.events import ArrivalEvent, CompletionEvent, ContainerCleanupEvent, ContainerReadyEvent, ScheduledEvent
from sim.model.events.region_delivery_event import RegionDeliveryEvent
from sim.model.events.registry_refresh_event import RegistryRefreshEvent
from sim.model.infrastructure import Infrastructure
from sim.model.load_balancer import build_load_balancer
from sim.model.node import Node
from sim.model.offloading import NodeScheduling, SchedulingPolicy
from sim.model.region import Region
from sim.model.request import Request, RequestStatus
from sim.observer import (
    ContainerStarted, ContainerStopped, FinalStatistics, SimulationError,
)
from sim.observer.observer import _Observer, observer as shared_observer
from sim.utils.validation import nonnegative_integer

from .engine import SimulationEngine
from .metrics import SimulationMetrics


@dataclass(frozen=True)
class SimulationContext:
    """Contesto condiviso della simulazione.

    Fornisce accesso al motore, all'infrastruttura, alle metriche della
    replica e all'observer condiviso.
    """

    scenario: Scenario
    simulation: 'Simulation'
    regions: Mapping[str, Region]
    metrics: SimulationMetrics
    observer: _Observer = field(default=shared_observer, init=False, repr=False, compare=False)
    engine: SimulationEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Inizializza il motore con il contesto.
        object.__setattr__(self, 'engine', SimulationEngine(self))

    @property
    def infrastructure(self) -> Infrastructure | None:
        """Infrastruttura della regione primaria."""
        return self.regions[self.scenario.primary_region].infrastructure


class Simulation:
    """Configura il motore e schedula gli arrivi iniziali."""

    def __init__(self, scenario: Scenario) -> None:

        # Inizializza la simulazione con uno scenario e genera un identificatore univoco per la replica.
        self.run_id = uuid4().hex

        # Costruisce il contesto della simulazione, che include l'infrastruttura e le metriche.
        self._ctx = SimulationContext(
            scenario=scenario,
            simulation=self,
            regions=self._build_regions(scenario),
            metrics=SimulationMetrics(),
        )
        invocation = scenario.invocation
        self._entry_region = self.regions[invocation.entry_region or scenario.primary_region]
        self._entry_node = None
        if invocation.entry_node is not None:
            infrastructure = self._entry_region.infrastructure
            assert infrastructure is not None
            self._entry_node = infrastructure.resolve_node(invocation.entry_node, area=invocation.entry_area)

        # Flag per evitare di avviare la simulazione più volte
        self._started = False
        self._ctx.observer.bind_context(self._ctx)

    @property
    def regions(self) -> Mapping[str, Region]:
        return self._ctx.regions

    def _build_regions(self, scenario: Scenario) -> Mapping[str, Region]:
        regions = {}
        for name, config in scenario.regions.items():
            infrastructure = (
                self._build_infrastructure(scenario, name) if config.infrastructure is not None else None
            )
            regions[name] = Region(
                name, infrastructure, config.service_time_s, config.offload_latency_s,
                config.offload_latency_by_source_s,
            )
        return MappingProxyType(regions)

    def _build_infrastructure(self, scenario: Scenario, name: str) -> Infrastructure:
        """Crea ring e nodi della singola regione."""
        config = scenario.regions[name].infrastructure
        assert config is not None
        stream_id = 'load_balancer' if name == scenario.primary_region else f'load_balancer/{name}'
        load_balancer = build_load_balancer(scenario.load_balancer, seed=scenario.seed, stream_id=stream_id)
        scheduler = scenario.scheduler
        infrastructure = Infrastructure(
            load_balancer, offloading_cache_validity=scheduler.offloading_cache_validity,
            load_balancer_areas=config.load_balancer_areas, forward_region=self._forward_region,
        )
        for ring in config.rings:
            infrastructure.create_ring(
                ring.id, arch=ring.arch, memory=ring.memory_mb, cpus=ring.cpus, replicas=ring.replicas,
            )
            for node in ring.nodes:
                infrastructure.create_node(
                    ring.id, node.name, node.area, node.machine_tag,
                    idle_timeout=scenario.containers.idle_timeout,
                    queue_capacity=scenario.scheduler.queue_capacity,
                )
        for ring in config.rings:
            live_ring = infrastructure.rings[ring.id]
            for node, node_config in zip(live_ring, ring.nodes, strict=True):
                neighbors = None
                if node_config.nearest_neighbors is not None:
                    neighbors = tuple(
                        infrastructure.resolve_node(peer, area=node_config.area) for peer in node_config.nearest_neighbors
                    )
                infrastructure.configure_scheduler(node, NodeScheduling(
                    policy=node_config.policy or scheduler.policy,
                    fallback_local=(
                        scheduler.fallback_local if node_config.fallback_local is None else node_config.fallback_local
                    ),
                    remote_area=node_config.remote_area,
                    nearest_neighbors=neighbors,
                    remote_region=node_config.remote_region,
                ))
        return infrastructure

    def submit(self, request: Request, *, region: str | None = None) -> Node | None:
        """Consegna un arrivo esterno alla regione indicata o all'ingresso dello scenario."""
        request.ensure_created()
        entry = self._entry_region if region is None else self.regions[region]
        node = self._entry_node if region is None else None
        area = self._ctx.scenario.invocation.entry_area if region is None else None
        self._ctx.metrics.record_external_arrival(entry.name)
        self._ctx.observer.notify_arrival(request, region=entry.name)
        destination = entry.submit(request, now=self._ctx.engine.now, node=node, area=area)
        self._record_submission(request)
        return destination

    def _forward_region(self, request: Request, name: str) -> Node | None:
        destination = self.regions[name]
        source = request.region
        assert source is not None
        self._ctx.metrics.record_offload(source, name)
        delay = destination.offload_latency_from(source) / 2
        request.start_transfer(name, delay)
        if delay == 0:
            return destination.submit(request, now=self._ctx.engine.now)
        self._ctx.engine.schedule_after(delay, RegionDeliveryEvent(request, destination, ctx=self._ctx))
        return None

    def deliver_region(self, request: Request, region: Region) -> None:
        now = self._ctx.engine.now
        self._ctx.metrics.record_transfer(request, now, -1)
        region.submit(request, now=now)
        self._record_submission(request)

    def _record_submission(self, request: Request) -> None:
        ctx = self._ctx
        now = ctx.engine.now
        if request.status is RequestStatus.IN_TRANSIT:
            ctx.metrics.record_transfer(request, now, 1)
            return

        ctx.metrics.record_submission(request, now)
        if request.status is RequestStatus.ACCEPTED:
            ctx.observer.notify_request_accepted(request)
            self._schedule_execution(request)
        elif request.status is RequestStatus.REJECTED:
            ctx.observer.notify_request_rejected(request)

    def run(self, arrivals: int) -> FinalStatistics:
        """Avvia la replica e la esegue senza orizzonte, salvo un limite di arrivi."""
        arrivals = nonnegative_integer(arrivals, 'arrivals')

        # Avvia la simulazione pianificando i primi arrivi per ogni funzione.
        self._start()

        # Esegue il motore finché una delle condizioni
        # di terminazione non è soddisfatta:
        # - si raggiunge il numero di arrivi richiesto
        # - non ci sono più eventi da processare.
        while self._ctx.metrics.arrivals < arrivals and self._ctx.engine.step():
            pass

        return self._final_statistics()

    def _start(self) -> tuple[ScheduledEvent, ...]:
        """Pianifica un primo arrivo per ogni funzione, in ordine di configurazione."""
        if self._started:
            raise RuntimeError('Simulazione già avviata')
        # Una replica può essere costruita prima del suo effettivo avvio.
        self._ctx.observer.bind_context(self._ctx)

        # Inizializza l'infrastruttura e registra i nodi
        # per la raccolta delle metriche.
        #
        # Imposta le callback per il tracciamento dello stato dei container
        # e la pulizia dei container idle.
        for region in self.regions.values():
            if region.infrastructure is None:
                continue
            for ring_id, ring in region.infrastructure.rings.items():
                for node in ring:
                    self._ctx.metrics.register_node(node, self._ctx.engine.now)
                    self._bind_node_callbacks(node, region.name, ring_id)
            self._start_offloading_monitoring(region.name, region.infrastructure)

        # Pianifica il primo arrivo per ogni funzione registrata nello scenario.
        first_arrivals = []
        for function in self._ctx.scenario.function_registry.values():
            # La funzione deve avere un campionatore per generare arrivi e durate di servizio.
            profile = function.profile
            assert profile is not None

            # Crea una copia del campionatore per evitare conflitti tra le occorrenze di arrivo.
            f = replace(function, sampler=deepcopy(function.sampler))

            # Pianifica l'arrivo iniziale al tempo specificato nel profilo della funzione.
            event = ArrivalEvent(f, ctx=self._ctx)
            scheduled_event = self._ctx.engine.schedule(profile.first_arrival_time, event)
            first_arrivals.append(scheduled_event)

        self._started = True
        return tuple(first_arrivals)

    def _start_offloading_monitoring(self, name: str, infrastructure: Infrastructure) -> None:
        ctx = self._ctx
        scheduler = ctx.scenario.scheduler
        topology = ctx.scenario.regions[name].infrastructure
        assert topology is not None
        configs = [node for ring in topology.rings for node in ring.nodes]
        if any((node.policy or scheduler.policy) is SchedulingPolicy.EDGE_ONLY for node in configs):
            refresh = RegistryRefreshEvent(infrastructure.refresh_neighbors, scheduler.nearby_interval, ctx=ctx)
            ctx.engine.schedule_after(refresh.interval, refresh)
        if any(node.remote_area and (node.policy or scheduler.policy) in (
            SchedulingPolicy.EDGE_CLOUD, SchedulingPolicy.CLOUD_ONLY,
        ) for node in configs):
            infrastructure.refresh_remote_targets()
            refresh = RegistryRefreshEvent(
                infrastructure.refresh_remote_targets, scheduler.monitoring_interval, ctx=ctx,
            )
            ctx.engine.schedule_after(refresh.interval, refresh)

    def _bind_node_callbacks(self, node: Node, region: str, ring_id: str) -> None:
        """Associa le callback per gli eventi dei nodi."""
        node.on_container_state_change = partial(self._record_container_state, region, ring_id)
        node.on_container_idle = self._schedule_container_cleanup
        node.on_dequeue = self._request_dequeued

    def _final_statistics(self) -> FinalStatistics:
        """Prepara la notifica dopo il completamento di run()."""
        return FinalStatistics(
            run_id=self.run_id,
            sim_time_s=self._ctx.engine.now,
            outcome='completed',
            metrics=self.metrics_snapshot(),
            summary=str(self._ctx.metrics),
        )



    def _schedule_execution(self, request: Request) -> None:
        """Pianifica il servizio nella regione di destinazione."""
        if request.warm or request.node is None:
            duration = request.service_duration
            if duration is None:
                raise ValueError('La durata di servizio della richiesta non può essere None')

            completion = CompletionEvent(request, ctx=self._ctx)
            if request.node is None and duration == 0:
                completion.execute()
            else:
                self._ctx.engine.schedule_after(duration, completion)

        # Se la richiesta è "cold", pianifica un ContainerReadyEvent
        # dopo il tempo di cold start della funzione.
        else:
            cold_start_time = request.function.cold_start_time
            if cold_start_time is None:
                raise ValueError('Il tempo di cold start della funzione non può essere None')

            self._ctx.engine.schedule_after(cold_start_time, ContainerReadyEvent(request, ctx=self._ctx),)

    def _request_dequeued(self, request: Request) -> None:
        """Registra la rimozione della richiesta dalla coda e notifica l'osservatore."""
        self._ctx.metrics.record_dequeue(request, self._ctx.engine.now)
        self._ctx.observer.notify_request_accepted(request)
        self._schedule_execution(request)




    def notify_event_error(self, event: ScheduledEvent, error: Exception) -> None:
        """Pubblica il fallimento di un'occorrenza per questa replica."""
        self._ctx.observer.notify(SimulationError(
            run_id=self.run_id,
            sim_time_s=self._ctx.engine.now,
            failed_event=type(event.event).__name__,
            exception=error,
        ))

    def _record_container_state(
        self, region: str, ring_id: str, node: Node, container: DockerContainer, previous: ContainerState | None,
    ) -> None:
        now = self._ctx.engine.now
        self._ctx.metrics.record_container_state(node, container, previous, now)

        if previous is ContainerState.CREATING and container.state in (ContainerState.BUSY, ContainerState.IDLE):
            notification = ContainerStarted
        elif previous is ContainerState.IDLE and container.state is ContainerState.DESTROYED:
            notification = ContainerStopped
        else:
            return

        self._ctx.observer.notify(notification(
            self.run_id, now, ring_id, node.id.key, container.id, container.function.name, region=region,
        ))

    def _schedule_container_cleanup(self, node: Node, container: DockerContainer) -> None:
        """Pianifica il primo passaggio del janitor strettamente dopo la scadenza."""
        expiration = container.expiration_time
        assert expiration is not None  # Il nodo notifica solo container in idle.
        interval = self._ctx.scenario.containers.cleanup_interval
        deadline = max(expiration, self._ctx.engine.now)
        tick = floor(deadline / interval) + 1
        check_time = tick * interval
        if check_time <= deadline:
            check_time = (tick + 1) * interval
        if check_time <= deadline:
            raise ValueError('Intervallo di pulizia troppo piccolo per avanzare il tempo')
        self._ctx.engine.schedule(check_time, ContainerCleanupEvent(node, container, ctx=self._ctx))

    def metrics_snapshot(self) -> dict:
        """Raccoglie le metriche della replica al tempo corrente."""
        return self._ctx.metrics.snapshot(self._ctx.engine.now, self.regions, self._ctx.scenario.primary_region)
