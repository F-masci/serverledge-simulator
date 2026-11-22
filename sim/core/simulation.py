"""Coordinamento degli arrivi sul calendario della simulazione."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import floor
from types import MappingProxyType
from uuid import uuid4

from sim.bindings import NodeBindings, RequestBindings
from sim.io.scenario import Scenario
from sim.model.container import DockerContainer
from sim.model.events import (
    ArrivalEvent, CompletionEvent, ContainerCleanupEvent, ContainerReadyEvent,
    RegionDeliveryEvent, ResponseEvent, ScheduledEvent,
)
from sim.model.node import Node
from sim.model.offloading import NodeScheduling
from sim.model.region import Region
from sim.model.request import Request, RequestStatus
from sim.observer import FinalStatistics
from sim.observer.observer import Observer, observer as shared_observer
from sim.metrics import SimulationMetrics
from sim.utils.report import format_metrics
from sim.utils.validation import nonnegative_integer
from .engine import SimulationEngine


@dataclass(frozen=True)
class SimulationContext:
    """Contesto condiviso della simulazione.

    Fornisce accesso al motore, all'infrastruttura, alle metriche della
    replica e al suo observer.
    """

    scenario: Scenario
    simulation: 'Simulation'
    metrics: SimulationMetrics
    regions: Mapping[str, Region] = field(init=False)
    observer: Observer = field(init=False, repr=False, compare=False)
    engine: SimulationEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Inizializza il motore e l'observer prima della costruzione delle regioni,
        # così i binding applicati alla creazione dei nodi trovano il contesto pronto.
        object.__setattr__(self, 'engine', SimulationEngine(self))
        observer = Observer(self, parent=shared_observer)
        observer.subscribe(self.metrics.on_notification, propagate_errors=True)
        object.__setattr__(self, 'observer', observer)
        object.__setattr__(self, 'regions', MappingProxyType({}))

    @property
    def now(self) -> float:
        """Tempo simulato in secondi."""
        return self.engine.now


class Simulation:
    """Configura il motore e schedula gli arrivi iniziali."""

    def __init__(self, scenario: Scenario) -> None:

        # Inizializza la simulazione con uno scenario e genera un identificatore univoco per la replica.
        self.run_id = uuid4().hex

        # Costruisce il contesto della simulazione (motore, observer e metriche)
        # e popola le regioni con i binding di ciclo di vita dei nodi.
        self._ctx = SimulationContext(
            scenario=scenario,
            simulation=self,
            metrics=SimulationMetrics(),
        )
        object.__setattr__(self._ctx, 'regions', self._build_regions(scenario))

        # Determina la regione di ingresso e il nodo di ingresso,
        # se specificato nello scenario.
        #
        # La regione di ingresso è quella specificata nell'invocazione
        # dello scenario o, in mancanza, la regione primaria.
        # Il nodo di ingresso è quello specificato nell'invocazione, se presente.
        #
        # Se il nodo di ingresso non è specificato, la richiesta viene consegnata
        # alla regione di ingresso e lo scheduler della regione deciderà il nodo di destinazione.
        invocation = scenario.invocation
        self._entry_region: Region = self.regions[invocation.entry_region or scenario.primary_region]
        self._entry_node: Node | None = None
        if invocation.entry_node is not None:
            infrastructure = self._entry_region.infrastructure
            assert infrastructure is not None
            self._entry_node = infrastructure.resolve_node(invocation.entry_node)

        # Flag per evitare di avviare la simulazione più volte
        self._started = False
        
    ### Simulation ###
    
    @property
    def now(self) -> float:
        """Tempo simulato in secondi."""
        return self._ctx.now
        
    ### Infrastructure ###

    @property
    def regions(self) -> Mapping[str, Region]:
        return self._ctx.regions

    def _build_regions(self, scenario: Scenario) -> Mapping[str, Region]:
        """Costruisce tutte le regioni dello scenario con i binding dei nodi e configura gli scheduler."""
        node_bindings = self._build_node_bindings()
        regions = {
            name: Region(name, scenario=scenario, node_bindings=node_bindings)
            for name in scenario.regions
        }
        self._configure_schedulers(scenario, regions)
        return MappingProxyType(regions)

    @staticmethod
    def _configure_schedulers(scenario: Scenario, regions: Mapping[str, Region]) -> None:
        """Risolvi i riferimenti remoti dopo la costruzione di tutte le regioni."""
        scheduler = scenario.scheduler
        for name, region in regions.items():
            infrastructure = region.infrastructure
            config = scenario.regions[name].infrastructure
            if infrastructure is None or config is None:
                continue
            for ring in config.rings:
                live_ring = infrastructure.rings[ring.id]
                for node, node_config in zip(live_ring, ring.nodes, strict=True):
                    neighbors = None
                    if node_config.nearest_neighbors is not None:
                        neighbors = tuple(
                            infrastructure.resolve_node(peer) for peer in node_config.nearest_neighbors
                        )
                    infrastructure.configure_scheduler(node, NodeScheduling(
                        policy=node_config.policy or scheduler.policy,
                        fallback_local=(
                            scheduler.fallback_local if node_config.fallback_local is None else node_config.fallback_local
                        ),
                        nearest_neighbors=neighbors,
                        remote_region=regions[node_config.remote_region] if node_config.remote_region is not None else None,
                        offloading_cache_validity_s=scheduler.offloading_cache_validity_s,
                    ))

    #############################
    ### Simulation management ###
    #############################

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

        # Pianifica il primo arrivo per ogni funzione registrata nello scenario.
        request_bindings = self._build_request_bindings()
        first_arrivals = []
        for function in self._ctx.scenario.function_registry.values():
            # La funzione deve avere un campionatore per generare arrivi e durate di servizio.
            profile = function.profile
            assert profile is not None

            # Crea una copia del campionatore per evitare conflitti tra le occorrenze di arrivo.
            function = replace(function, sampler=deepcopy(function.sampler))

            # Pianifica l'arrivo iniziale al tempo specificato nel profilo della funzione.
            event = ArrivalEvent(
                ctx=self._ctx,
                function=function,
                entry_region=self._entry_region,
                entry_node=self._entry_node,
                request_bindings=request_bindings,
            )
            scheduled_event = self._ctx.engine.schedule(profile.first_arrival_time, event)
            first_arrivals.append(scheduled_event)

        self._started = True
        return tuple(first_arrivals)

    def _final_statistics(self) -> FinalStatistics:
        """Prepara la notifica dopo il completamento di run()."""
        metrics = self.metrics_snapshot()
        return FinalStatistics(
            run_id=self.run_id,
            sim_time_s=self.now,
            outcome='completed',
            metrics=metrics,
            summary=format_metrics(metrics),
        )


    def metrics_snapshot(self) -> dict:
        """Raccoglie le metriche della replica al tempo corrente."""
        return self._ctx.metrics.snapshot(self.now, self.regions, self._ctx.scenario.primary_region)

    ################
    ### Callback ###
    ################

    ## Handlers ##

    def _schedule_execution(self, request: Request) -> None:
        """
        Pianifica il servizio nella regione di destinazione.
        Se la richiesta è "warm", pianifica un CompletionEvent dopo la durata di servizio.
        Se la richiesta è "cold", pianifica un ContainerReadyEvent dopo il tempo di cold start della funzione.

        Callback invocata quando una richiesta viene accettata da un nodo.
        """
        if request.warm or request.node is None:
            duration = request.service_duration
            if duration is None:
                raise ValueError('La durata di servizio della richiesta non può essere None')

            completion = CompletionEvent(ctx=self._ctx, request=request)
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

            container_ready_event = ContainerReadyEvent(ctx=self._ctx, request=request)
            self._ctx.engine.schedule_after(cold_start_time, container_ready_event)

    def _on_request_state_changed(
        self, request: Request, previous: RequestStatus, current: RequestStatus,
    ) -> None:
        """Gestisce accettazione e ritorno dei rifiuti remoti della richiesta."""
        if current is RequestStatus.RESPONDING and previous is RequestStatus.CREATED:
            self._ctx.observer.notify_request_rejection_returning(request)
            self._ctx.engine.schedule_after(
                request.return_delay, ResponseEvent(ctx=self._ctx, request=request),
            )
            return
        if current is not RequestStatus.ACCEPTED:
            return

        # Calcola il tempo di attesa in coda solo se la richiesta proviene dalla FIFO.
        queue_wait = self.now - request.region_arrival_time if previous is RequestStatus.QUEUED else None
        self._ctx.observer.notify_request_accepted(request, queue_wait_s=queue_wait)
        self._schedule_execution(request)

    def _request_dequeued(self, node: Node, request: Request) -> None:
        """
        Notifica l'estrazione dalla coda FIFO del nodo.

        Callback invocata dal nodo prima dell'ammissione della richiesta sul container.
        """
        self._ctx.observer.notify_request_dequeued(request)


    def _schedule_container_cleanup(self, node: Node, container: DockerContainer) -> None:
        """
        Pianifica il primo passaggio del janitor strettamente dopo la scadenza.

        Callback invocata dal nodo quando un container entra nello stato idle.
        """
        expiration = container.expiration_time
        assert expiration is not None  # Il nodo notifica solo container in idle.
        interval = self._ctx.scenario.containers.cleanup_interval
        deadline = max(expiration, self.now)
        tick = floor(deadline / interval) + 1
        check_time = tick * interval
        if check_time <= deadline:
            check_time = (tick + 1) * interval
        if check_time <= deadline:
            raise ValueError('Intervallo di pulizia troppo piccolo per avanzare il tempo')
        self._ctx.engine.schedule(check_time, ContainerCleanupEvent(node, container, ctx=self._ctx))

    def _schedule_region_delivery(
        self, node: Node, request: Request, destination: Region, target: Node | None,
    ) -> Node | None:
        """
        Notifica l'inoltro verso la regione remota e pianifica la consegna.

        Se la latenza di sola andata è nulla consegna immediatamente la richiesta
        alla regione di destinazione; altrimenti pianifica un RegionDeliveryEvent.

        Callback invocata dal nodo al termine di _forward_region.
        """
        delay = request.return_delay
        self._ctx.observer.notify_request_forwarded(request, node.region)

        # Con latenza nulla la consegna avviene immediatamente nello stesso istante simulato.
        if delay == 0:
            self._ctx.observer.notify_request_delivered(request)
            return destination.submit(request, now=self.now, node=target)

        # Con latenza positiva pianifica l'evento di consegna alla regione remota.
        delivery_event = RegionDeliveryEvent(
            ctx=self._ctx, request=request, region=destination, node=target,
        )
        self._ctx.engine.schedule_after(delay, delivery_event)
        return None

    ## Binding ##

    def _build_node_bindings(self) -> NodeBindings:
        """Costruisce i binding di ciclo di vita per ogni nodo creato."""
        return NodeBindings(
            on_node_registered=self._ctx.observer.notify_node_registered,
            on_node_removed=self._ctx.observer.notify_node_removed,
            on_container_state_changed=self._ctx.observer.notify_container_state,
            on_container_idle=self._schedule_container_cleanup,
            on_request_dequeued=self._request_dequeued,
            on_request_forwarded=self._schedule_region_delivery,
        )

    def _build_request_bindings(self) -> RequestBindings:
        """Costruisce i binding di ciclo di vita applicati a ogni richiesta creata."""
        return RequestBindings(
            on_queued=self._ctx.observer.notify_request_queued,
            on_rejected=self._ctx.observer.notify_request_rejected,
            on_completed=self._ctx.observer.notify_request_completed,
            on_state_changed=self._on_request_state_changed,
        )
