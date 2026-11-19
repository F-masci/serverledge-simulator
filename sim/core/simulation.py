"""Coordinamento degli arrivi sul calendario della simulazione."""

from copy import deepcopy
from dataclasses import replace
from math import floor

from sim.io.scenario import Scenario
from sim.model.container import ContainerState, DockerContainer
from sim.model.events import ArrivalEvent, ContainerCleanupEvent, ScheduledEvent
from sim.model.infrastructure import Infrastructure
from sim.model.node import Node

from .engine import SimulationEngine
from .metrics import SimulationMetrics


class Simulation:
    """Configura il motore e schedula gli arrivi iniziali."""

    def __init__(self, scenario: Scenario, infrastructure: Infrastructure | None = None) -> None:
        self._scenario = scenario
        if infrastructure is None:
            infrastructure = scenario.build_infrastructure()
        self._engine = SimulationEngine(simulation=self, infrastructure=infrastructure)
        self._infrastructure = infrastructure
        self.metrics = SimulationMetrics()
        self._started = False

    @property
    def engine(self) -> SimulationEngine:
        return self._engine

    @property
    def arrivals(self) -> int:
        """Numero di richieste sottoposte all'infrastruttura."""
        return self.metrics.arrivals

    def start(self) -> tuple[ScheduledEvent, ...]:
        """Pianifica un primo arrivo per ogni funzione, in ordine di configurazione."""
        if self._started:
            raise RuntimeError('Simulazione già avviata')
        for ring in self._infrastructure.rings.values():
            for node in ring:
                self.metrics.register_node(node, self.engine.now)
                node.on_container_state_change = self._record_container_state
                node.on_container_idle = self._schedule_container_cleanup
                # I nodi creati dallo scenario partono vuoti: il primo evento di
                # pulizia viene pianificato dalla callback alla transizione a IDLE.
                # Qui recuperiamo solo eventuali container già idle se viene
                # passata un'infrastruttura con pool già popolati.
                for pool in node.pools.values():
                    for container in pool.idle:
                        self._schedule_container_cleanup(node, container)
        first_arrivals = []
        for function in self._scenario.function_registry.values():
            profile = function.profile
            assert profile is not None  # Scenario valida il profilo alla registrazione.
            replica_function = replace(function, sampler=deepcopy(function.sampler))
            event = ArrivalEvent(replica_function)
            first_arrivals.append(self.engine.schedule(profile.first_arrival_time, event))
        self._started = True
        return tuple(first_arrivals)

    def _record_container_state(
        self, node: Node, container: DockerContainer, previous: ContainerState | None,
    ) -> None:
        self.metrics.record_container_state(node, container, previous, self.engine.now)

    def _schedule_container_cleanup(self, node: Node, container: DockerContainer) -> None:
        """Pianifica il primo passaggio del janitor strettamente dopo la scadenza."""
        expiration = container.expiration_time
        assert expiration is not None  # Il nodo notifica solo container in idle.
        interval = self._scenario.container_cleanup_interval
        deadline = max(expiration, self.engine.now)
        tick = floor(deadline / interval) + 1
        check_time = tick * interval
        if check_time <= deadline:
            check_time = (tick + 1) * interval
        if check_time <= deadline:
            raise ValueError('Intervallo di pulizia troppo piccolo per avanzare il tempo')
        self.engine.schedule(check_time, ContainerCleanupEvent(node, container))

    def metrics_snapshot(self) -> dict:
        """Raccoglie le metriche della replica al tempo corrente."""
        return self.metrics.snapshot(self.engine.now, self._infrastructure)
