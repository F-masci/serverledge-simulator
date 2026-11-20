"""Coordinamento degli arrivi sul calendario della simulazione."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from functools import partial
from math import floor
from uuid import uuid4

from sim.io.scenario import Scenario
from sim.model.container import ContainerState, DockerContainer
from sim.model.events import ArrivalEvent, CompletionEvent, ContainerCleanupEvent, ContainerReadyEvent, ScheduledEvent
from sim.model.infrastructure import Infrastructure
from sim.model.node import Node
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
    infrastructure: Infrastructure
    metrics: SimulationMetrics
    observer: _Observer = field(default=shared_observer, init=False, repr=False, compare=False)
    engine: SimulationEngine = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Inizializza il motore con il contesto.
        object.__setattr__(self, 'engine', SimulationEngine(self))


class Simulation:
    """Configura il motore e schedula gli arrivi iniziali."""

    def __init__(self, scenario: Scenario) -> None:

        # Inizializza la simulazione con uno scenario e genera un identificatore univoco per la replica.
        self.run_id = uuid4().hex

        # Costruisce il contesto della simulazione, che include l'infrastruttura e le metriche.
        self._ctx = SimulationContext(
            scenario=scenario,
            simulation=self,
            infrastructure=scenario.build_infrastructure(),
            metrics=SimulationMetrics(),
        )

        # Flag per evitare di avviare la simulazione più volte
        self._started = False
        self._ctx.observer.bind_context(self._ctx)

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
        for ring_id, ring in self._ctx.infrastructure.rings.items():
            for node in ring:
                self._ctx.metrics.register_node(node, self._ctx.engine.now)
                self._bind_node_callbacks(node, ring_id)

        # Pianifica il primo arrivo per ogni funzione registrata nello scenario.
        first_arrivals = []
        for function in self._ctx.scenario.function_registry.values():
            # La funzione deve avere un campionatore per generare arrivi e durate di servizio.
            profile = function.profile
            assert profile is not None

            # Crea una copia del campionatore per evitare conflitti tra le occorrenze di arrivo.
            f = replace(function, sampler=deepcopy(function.sampler))

            # Pianifica l'arrivo iniziale al tempo specificato nel profilo della funzione.
            event = ArrivalEvent(f, on_accepted=self._schedule_execution, ctx=self._ctx)
            scheduled_event = self._ctx.engine.schedule(profile.first_arrival_time, event)
            first_arrivals.append(scheduled_event)

        self._started = True
        return tuple(first_arrivals)

    def _bind_node_callbacks(self, node: Node, ring_id: str) -> None:
        """Associa le callback per gli eventi dei nodi."""
        node.on_container_state_change = partial(self._record_container_state, ring_id)
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
        """Pianifica l'esecuzione della richiesta sul nodo assegnato."""

        # Se la richiesta è "warm", pianifica un CompletionEvent dopo la durata di servizio.
        if request.warm:
            duration = request.service_duration
            if duration is None:
                raise ValueError('La durata di servizio della richiesta non può essere None')

            self._ctx.engine.schedule_after(duration, CompletionEvent(request, ctx=self._ctx))

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
        self, ring_id: str, node: Node, container: DockerContainer, previous: ContainerState | None,
    ) -> None:
        now = self._ctx.engine.now
        self._ctx.metrics.record_container_state(node, container, previous, now)

        if previous is ContainerState.CREATING and container.state in (ContainerState.BUSY, ContainerState.IDLE):
            notification = ContainerStarted
        elif previous is ContainerState.IDLE and container.state is ContainerState.DESTROYED:
            notification = ContainerStopped
        else:
            return

        self._ctx.observer.notify(notification(self.run_id, now, ring_id, node.id.key, container.id, container.function.name))

    def _schedule_container_cleanup(self, node: Node, container: DockerContainer) -> None:
        """Pianifica il primo passaggio del janitor strettamente dopo la scadenza."""
        expiration = container.expiration_time
        assert expiration is not None  # Il nodo notifica solo container in idle.
        interval = self._ctx.scenario.container_cleanup_interval
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
        return self._ctx.metrics.snapshot(self._ctx.engine.now, self._ctx.infrastructure)
