"""Definizione delle funzioni e dei profili di simulazione."""

import json
from dataclasses import dataclass, field

from simulator.random.distributions import (
    ArrivalProcess, DeterministicArrivalProcess, DeterministicServiceTime,
    GammaServiceTime, PoissonArrivalProcess, ServiceTime,
)
from simulator.random.streams import RandomStreams
from simulator.utils.machine_tag import matches_machine_tag, validate_tag_pattern
from simulator.utils.validation import integer, nonnegative_number, one_of, positive_integer, positive_number

from .runtime import CustomRuntime, Runtime


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

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile del profilo di simulazione."""
        return (
            f'FunctionProfile(arrival_rate={self.arrival_rate}, '
            f'service_rate={self.service_rate}, '
            f'arrival_distribution={self.arrival_distribution}, '
            f'service_distribution={self.service_distribution})'
        )

    def to_dict(self) -> dict:
        """Restituisce il profilo di simulazione in formato serializzabile."""
        return {
            'arrival_rate': self.arrival_rate,
            'service_rate': self.service_rate,
            'first_arrival_time': self.first_arrival_time,
            'arrival_distribution': self.arrival_distribution,
            'service_distribution': self.service_distribution,
            'service_scv': self.service_scv,
            'code_load_time': self.code_load_time,
        }

    def to_json(self) -> str:
        """Serializza il profilo per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)


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
        if not self.name:
            raise ValueError('Nome funzione obbligatorio')
        positive_integer(self.memory_mb, 'function.memory_mb')
        positive_number(self.cpu_demand, 'function.cpu_demand')
        integer(self.max_concurrency, 'function.max_concurrency')
        if self.supported_archs is None and isinstance(self.runtime, CustomRuntime):
            raise ValueError('Specificare le architetture risolte per la immagine custom')
        # Nota: nell'API Go (internal/api/api.go) i runtime integrati ricevono sempre
        # ('amd64', 'arm64'); il simulatore consente di restringerli nello scenario per
        # modellare vincoli architetturali senza dover dichiarare immagini custom.
        archs = ('amd64', 'arm64') if self.supported_archs is None else tuple(self.supported_archs)
        if not archs:
            raise ValueError('supported_archs non può essere vuoto')
        validate_tag_pattern(self.tag_pattern)
        object.__setattr__(self, 'supported_archs', archs)
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

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        """Restituisce la configurazione della funzione in formato serializzabile."""
        return {
            'name': self.name,
            'memory_mb': self.memory_mb,
            'cpu_demand': self.cpu_demand,
            'runtime': self.runtime.to_dict(),
            'max_concurrency': self.max_concurrency,
            'supported_archs': list(self.supported_archs) if self.supported_archs is not None else None,
            'tag_pattern': self.tag_pattern,
            'tar_function_code': self.tar_function_code,
            'cold_start_time_s': self.cold_start_time,
            'profile': self.profile.to_dict() if self.profile is not None else None,
        }

    def to_json(self) -> str:
        """Serializza la funzione per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
