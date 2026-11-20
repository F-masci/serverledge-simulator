"""Definizione delle funzioni e politiche di concorrenza dei runtime."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import ClassVar

from sim.random.distributions import (
    ArrivalProcess, DeterministicArrivalProcess, DeterministicServiceTime,
    GammaServiceTime, PoissonArrivalProcess, ServiceTime,
)
from sim.random.streams import RandomStreams
from sim.utils.validation import finite_time, nonempty_string


_BUILTIN_IMAGES = {
    'python314': 'grussorusso/serverledge-python314',
    'python312ml': 'grussorusso/serverledge-python312ml',
    'go125': 'grussorusso/serverledge-go125',
    'nodejs17ng': 'grussorusso/serverledge-nodejs17ng',
}


@dataclass(frozen=True, kw_only=True)
class Runtime(ABC):
    """Politica di normalizzazione della concorrenza alla registrazione."""

    _runtimes: ClassVar[dict[tuple[str, str, float], 'Runtime']] = {}
    startup_time: float = 0.0

    @staticmethod
    def get_runtime(
        name: str, *, custom_image: str | None = None, startup_time: float = 0.0,
        startup_times: Mapping[str, float] | None = None,
    ) -> 'Runtime':
        """Restituisce l'istanza condivisa per runtime, immagine e tempo di avvio."""

        # Cerca il nome tra i runtime integrati e risolve l'immagine Docker.
        builtin = name in _BUILTIN_IMAGES
        if builtin:
            if custom_image is not None:
                raise ValueError('custom_image è ammesso solo per il runtime custom')
            image = _BUILTIN_IMAGES[name]
        else:
            # Un nome non integrato può indicare direttamente un'immagine custom.
            if name == 'custom' and custom_image is None:
                raise ValueError('Immagine custom obbligatoria')
            image = nonempty_string(custom_image if custom_image is not None else name, 'custom_image')

        # Usa il tempo configurato per l'immagine, se presente nello scenario.
        if startup_times is not None:
            startup_time = startup_times.get(image, startup_time)
        startup_time = finite_time(startup_time, 'runtime.startup_time')

        # Per i custom l'identità dipende dall'immagine, anche se dichiarata in forme diverse.
        key = (name if builtin else 'custom', image, startup_time)
        if key in Runtime._runtimes:
            return Runtime._runtimes[key]

        # Costruisce e registra il runtime solo alla prima richiesta.
        if builtin:
            runtime = BuiltinRuntime(name, startup_time=startup_time)
        else:
            runtime = CustomRuntime(image, startup_time=startup_time)
        return Runtime._runtimes.setdefault(key, runtime)

    def __post_init__(self) -> None:
        if finite_time(self.startup_time, 'runtime.startup_time') < 0:
            raise ValueError('runtime.startup_time deve essere non negativo')

    @abstractmethod
    def effective_concurrency(self, requested: int) -> int:
        """Restituisce il limite applicabile ai container della funzione."""


@dataclass(frozen=True)
class BuiltinRuntime(Runtime):
    name: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name not in _BUILTIN_IMAGES:
            raise ValueError(f'Runtime sconosciuto: {self.name}')

    @property
    def image(self) -> str:
        return _BUILTIN_IMAGES[self.name]

    def effective_concurrency(self, requested: int) -> int:
        return 1 if self.name == 'nodejs17ng' else max(1, requested)


@dataclass(frozen=True)
class CustomRuntime(Runtime):
    image: str

    def __post_init__(self) -> None:
        super().__post_init__()
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
    code_load_time: float = 0.0

    def __post_init__(self) -> None:
        if finite_time(self.code_load_time, 'function.code_load_time') < 0:
            raise ValueError('function.code_load_time deve essere non negativo')
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

    @property
    def cold_start_time(self) -> float:
        """Somma il tempo di avvio dell'immagine e il caricamento del codice."""
        return self.runtime.startup_time + (self.profile.code_load_time if self.profile else 0.0)

    def supports_arch(self, arch: str) -> bool:
        return arch in self.supported_archs

    def __str__(self) -> str:
        return self.name
