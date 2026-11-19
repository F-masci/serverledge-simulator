"""Coordinamento degli arrivi sul calendario della simulazione."""

from copy import deepcopy
from dataclasses import replace

from sim.io.scenario import Scenario
from sim.model.events import ArrivalEvent, ScheduledEvent
from sim.model.infrastructure import Infrastructure

from .engine import SimulationEngine


class Simulation:
    """Configura il motore e schedula gli arrivi iniziali."""

    def __init__(self, scenario: Scenario, infrastructure: Infrastructure | None = None) -> None:
        self._scenario = scenario
        if infrastructure is None:
            infrastructure = Infrastructure()
        self._engine = SimulationEngine(simulation=self, infrastructure=infrastructure)
        self._arrivals = 0
        self._started = False

    @property
    def engine(self) -> SimulationEngine:
        return self._engine

    @property
    def arrivals(self) -> int:
        """Numero di richieste sottoposte all'infrastruttura."""
        return self._arrivals

    def start(self) -> tuple[ScheduledEvent, ...]:
        """Pianifica un primo arrivo per ogni funzione, in ordine di configurazione."""
        if self._started:
            raise RuntimeError('Simulazione già avviata')
        first_arrivals = []
        for function in self._scenario.function_registry.values():
            profile = function.profile
            assert profile is not None  # Scenario valida il profilo alla registrazione.
            replica_function = replace(function, sampler=deepcopy(function.sampler))
            event = ArrivalEvent(replica_function)
            first_arrivals.append(self.engine.schedule(profile.first_arrival_time, event))
        self._started = True
        return tuple(first_arrivals)

    def record_arrival(self) -> None:
        """Conta una richiesta sottoposta all'infrastruttura."""
        self._arrivals += 1
