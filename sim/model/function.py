"""Definizione delle funzioni e dei profili di simulazione."""

from dataclasses import dataclass, field

from sim.random.distributions import (
    ArrivalProcess, DeterministicArrivalProcess, DeterministicServiceTime,
    GammaServiceTime, PoissonArrivalProcess, ServiceTime,
)
from sim.random.streams import RandomStreams
from sim.utils.machine_tag import matches_machine_tag
from sim.utils.validation import nonnegative_number, one_of, positive_number

from .runtime import BuiltinRuntime as BuiltinRuntime, CustomRuntime, Runtime


@dataclass(frozen=True)
class FunctionSampler:
    """Campionatori di arrivo e servizio associati a una funzione registrata."""

    arrivals: ArrivalProcess
    service: ServiceTime


@dataclass(frozen=True)
class FunctionProfile:
    """Tassi di arrivo e servizio della funzione simulata, in richieste al secondo."""

    arrival_rate: float
    service_rate: float
    first_arrival_time: float
    arrival_distribution: str = 'poisson'
    service_distribution: str = 'gamma'
    service_scv: float | None = None
    code_load_time: float = 0.0

    def __post_init__(self) -> None:
        nonnegative_number(self.code_load_time, 'function.code_load_time')
        nonnegative_number(self.first_arrival_time, 'function.first_arrival_time')
        positive_number(self.arrival_rate, 'function.arrival_rate')
        positive_number(self.service_rate, 'function.service_rate')

        one_of(self.arrival_distribution, ('poisson', 'deterministic'), 'arrival_distribution')
        one_of(self.service_distribution, ('exponential', 'gamma', 'deterministic'), 'service_distribution')
        if self.service_scv is not None:
            if self.service_distribution != 'gamma':
                raise ValueError('service_scv è ammesso solo per il servizio gamma')
            positive_number(self.service_scv, 'function.service_scv')

    def build_sampler(
        self, name: str, streams: RandomStreams,
    ) -> FunctionSampler:
        """Costruisce i campionatori del profilo per la registrazione."""
        arrivals: ArrivalProcess
        if self.arrival_distribution == 'deterministic':
            arrivals = DeterministicArrivalProcess(self.arrival_rate)
        else:
            arrivals = PoissonArrivalProcess(self.arrival_rate, streams, stream_id=name)

        service: ServiceTime
        mean = 1 / self.service_rate
        if self.service_distribution == 'deterministic':
            service = DeterministicServiceTime(mean)
        else:
            scv = self.service_scv if self.service_scv is not None else 1.0
            service = GammaServiceTime(mean, scv, streams, stream_id=name)
        return FunctionSampler(arrivals, service)


@dataclass(frozen=True)
class Function:
    """Configurazione della funzione con un eventuale profilo di simulazione.

    supported_archs contiene la compatibilità già risolta alla registrazione;
    per immagini custom deve essere fornita dal chiamante.
    """

    name: str
    memory_mb: int
    cpu_demand: float
    runtime: Runtime
    max_concurrency: int = 1
    supported_archs: tuple[str, ...] | None = None
    tag_pattern: str = ''
    tar_function_code: str = ''
    profile: FunctionProfile | None = None
    sampler: FunctionSampler | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not self.name or self.memory_mb < 1:
            raise ValueError('Nome e memoria positiva obbligatori')
        positive_number(self.cpu_demand, 'function.cpu_demand')
        if self.supported_archs is None and isinstance(self.runtime, CustomRuntime):
            raise ValueError('Specificare le architetture risolte per la immagine custom')
        archs = ('amd64', 'arm64') if self.supported_archs is None else self.supported_archs
        object.__setattr__(self, 'supported_archs', tuple(archs))
        object.__setattr__(self, 'max_concurrency',
                           self.runtime.effective_concurrency(self.max_concurrency))

    @property
    def cold_start_time(self) -> float:
        """Somma il tempo di avvio dell'immagine e il caricamento del codice."""
        return self.runtime.startup_time + (self.profile.code_load_time if self.profile else 0.0)

    def supports_arch(self, arch: str) -> bool:
        return arch in self.supported_archs

    def supports_machine_tag(self, machine_tag: str) -> bool:
        return matches_machine_tag(machine_tag, self.tag_pattern)

    def __str__(self) -> str:
        return self.name
