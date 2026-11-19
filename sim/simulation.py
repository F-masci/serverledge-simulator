"""Coordinamento dell'avvio della simulazione."""

from sim.core import Event, ScheduledEvent, SimulationEngine
from .io.scenario import ScenarioConfig


# TODO: rimuovere questo segnaposto quando l'avvio schedulerà il primo arrivo.
class StartupEvent(Event):
    def execute(self, engine: SimulationEngine) -> None:
        """Segna il primo punto del calendario."""


class Simulation:
    """Possiede il motore e pianifica l'avvio a partire dai parametri dello scenario."""

    def __init__(self, scenario: ScenarioConfig) -> None:
        self._scenario = scenario
        self._engine = SimulationEngine()
        self._started = False

    @property
    def engine(self) -> SimulationEngine:
        return self._engine

    def start(self) -> ScheduledEvent:
        """Inserisce il primo evento nel calendario a orizzonte aperto."""
        if self._started:
            raise RuntimeError('Simulazione già avviata')
        first_event = self.engine.schedule(self._scenario.first_event_time, StartupEvent())
        self._started = True
        return first_event
