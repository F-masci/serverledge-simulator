"""Definizione delle funzioni e politiche di concorrenza dei runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math

from sim.random.distributions import (
    ArrivalProcess, DeterministicArrivalProcess, DeterministicServiceTime,
    GammaServiceTime, PoissonArrivalProcess, ServiceTime,
)
from sim.random.streams import RandomStreams
from sim.utils.validation import finite_time


class Runtime(ABC):
    """Politica di normalizzazione della concorrenza alla registrazione."""

    @abstractmethod
    def effective_concurrency(self, requested: int) -> int:
        """Restituisce il limite applicabile ai container della funzione."""


@dataclass(frozen=True)
class BuiltinRuntime(Runtime):
    name: str

    def __post_init__(self) -> None:
        if self.name not in {'python314', 'python312ml', 'go125', 'nodejs17ng'}:
            raise ValueError(f'Runtime sconosciuto: {self.name}')

    def effective_concurrency(self, requested: int) -> int:
        return 1 if self.name == 'nodejs17ng' else max(1, requested)


@dataclass(frozen=True)
class CustomRuntime(Runtime):
    image: str

    def __post_init__(self) -> None:
        if not self.image:
            raise ValueError('Immagine custom obbligatoria')

    def effective_concurrency(self, requested: int) -> int:
        return 1


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

    def __post_init__(self) -> None:
        if finite_time(self.first_arrival_time, 'function.first_arrival_time') < 0:
            raise ValueError('function.first_arrival_time deve essere non negativo')
        if finite_time(self.arrival_rate, 'function.arrival_rate') <= 0:
            raise ValueError('function.arrival_rate deve essere positivo')
        if finite_time(self.service_rate, 'function.service_rate') <= 0:
            raise ValueError('function.service_rate deve essere positivo')

        if self.arrival_distribution not in ('poisson', 'deterministic'):
            raise ValueError('arrival_distribution deve essere poisson o deterministic')
        if self.service_distribution not in ('exponential', 'gamma', 'deterministic'):
            raise ValueError('service_distribution deve essere exponential, gamma o deterministic')
        if self.service_scv is not None:
            if self.service_distribution != 'gamma':
                raise ValueError('service_scv è ammesso solo per il servizio gamma')
            if finite_time(self.service_scv, 'service_scv') <= 0:
                raise ValueError('service_scv deve essere positivo')

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
        if not math.isfinite(self.cpu_demand) or self.cpu_demand <= 0:
            raise ValueError('CPU demand deve essere finita e positiva')
        if self.supported_archs is None and isinstance(self.runtime, CustomRuntime):
            raise ValueError('Specificare le architetture risolte per la immagine custom')
        archs = ('amd64', 'arm64') if self.supported_archs is None else self.supported_archs
        object.__setattr__(self, 'supported_archs', tuple(archs))
        object.__setattr__(self, 'max_concurrency',
                           self.runtime.effective_concurrency(self.max_concurrency))

    def supports_arch(self, arch: str) -> bool:
        return arch in self.supported_archs

    def __str__(self) -> str:
        return self.name
