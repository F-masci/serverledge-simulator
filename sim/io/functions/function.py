"""Configurazione dichiarativa delle funzioni lette dallo scenario.

Esempio TOML nello scenario:

    [[functions]]
    name = "inference"                    # obbligatorio (univoco)
    runtime = "python314"                 # obbligatorio (builtin o "custom")
    memory_mb = 128                       # obbligatorio (intero > 0, in MiB)
    cpu_demand = 1.0                      # obbligatorio (numero > 0)
    first_arrival_time = 0.5              # obbligatorio (secondi >= 0)
    arrival_rate = 3.0                    # obbligatorio (req/s > 0)
    service_rate = 2.0                    # obbligatorio (req/s > 0)

    custom_image = "myrepo/img:latest"    # opzionale (richiesto se runtime = "custom")
    max_concurrency = 1                   # opzionale (default: 1)
    supported_archs = ["amd64", "arm64"]  # opzionale (default: tutte le architetture)
    tag_pattern = "amd64*"                # opzionale (default: "")
    tar_function_code = ""                # opzionale (default: "")
    arrival_distribution = "poisson"      # opzionale (default: "poisson" | "deterministic" | ...)
    service_distribution = "gamma"        # opzionale (default: "gamma" | "deterministic" | ...)
    service_scv = 1.0                     # opzionale (SCV per distribuzioni che lo richiedono)
    code_load_time = 0.0                  # opzionale (default: 0.0 s)
"""

from collections.abc import Mapping
from dataclasses import dataclass

from sim.model.function import Function, FunctionProfile
from sim.model.runtime import Runtime
from sim.utils.validation import nonempty_string, positive_integer, positive_number


@dataclass(frozen=True)
class _FunctionConfig:
    """Definizione della funzione letta dallo scenario prima della registrazione."""

    name: str
    runtime: Runtime
    memory_mb: int
    cpu_demand: float
    max_concurrency: int
    supported_archs: tuple[str, ...] | None
    profile: FunctionProfile
    tag_pattern: str
    tar_function_code: str

    @classmethod
    def parse(
        cls, data: object, *, startup_times: Mapping[str, float] | None = None,
    ) -> '_FunctionConfig':
        """Parse della configurazione di una funzione."""
        required = {
            'name', 'runtime', 'memory_mb', 'cpu_demand', 'arrival_rate', 'service_rate', 'first_arrival_time',
        }
        optional = {
            'custom_image', 'max_concurrency', 'supported_archs', 'tag_pattern',
            'tar_function_code', 'arrival_distribution', 'service_distribution', 'service_scv', 'code_load_time',
        }
        if not isinstance(data, Mapping):
            raise ValueError('La definizione della funzione deve essere una tabella')
        if missing := required - data.keys():
            raise ValueError(f'Campi obbligatori mancanti: {sorted(missing)}')
        if unknown := data.keys() - required - optional:
            raise ValueError(f'Campi sconosciuti: {sorted(map(str, unknown))}')

        runtime_name = nonempty_string(data['runtime'], 'runtime')
        custom_image = nonempty_string(data['custom_image'], 'custom_image') if 'custom_image' in data else None
        runtime = Runtime.get_runtime(runtime_name, custom_image=custom_image, startup_times=startup_times)

        supported_archs = None
        if 'supported_archs' in data:
            archs = data['supported_archs']
            if not isinstance(archs, list) or not archs:
                raise ValueError('supported_archs deve essere una lista non vuota')
            supported_archs = tuple(nonempty_string(arch, 'supported_archs') for arch in archs)

        metadata = {}
        for key in ('tag_pattern', 'tar_function_code'):
            if key in data:
                if not isinstance(data[key], str):
                    raise ValueError(f'{key} deve essere una stringa')
                metadata[key] = data[key]

        name = nonempty_string(data['name'], 'name')
        memory_mb = positive_integer(data['memory_mb'], 'memory_mb')
        cpu_demand = positive_number(data['cpu_demand'], 'cpu_demand')
        max_concurrency = positive_integer(data.get('max_concurrency', 1), 'max_concurrency')

        profile = FunctionProfile(
            arrival_rate=data['arrival_rate'],
            service_rate=data['service_rate'],
            first_arrival_time=data['first_arrival_time'],
            arrival_distribution=data.get('arrival_distribution', 'poisson'),
            service_distribution=data.get('service_distribution', 'gamma'),
            service_scv=data.get('service_scv'),
            code_load_time=data.get('code_load_time', 0),
        )

        return cls(
            name=name,
            runtime=runtime,
            memory_mb=memory_mb,
            cpu_demand=cpu_demand,
            max_concurrency=max_concurrency,
            supported_archs=supported_archs,
            profile=profile,
            tag_pattern=metadata.get('tag_pattern', ''),
            tar_function_code=metadata.get('tar_function_code', ''),
        )

    def build_function(self) -> Function:
        """Costruisce il modello con le relative regole di ammissione."""
        return Function(
            name=self.name,
            runtime=self.runtime,
            memory_mb=self.memory_mb,
            cpu_demand=self.cpu_demand,
            max_concurrency=self.max_concurrency,
            supported_archs=self.supported_archs,
            profile=self.profile,
            tag_pattern=self.tag_pattern,
            tar_function_code=self.tar_function_code,
        )
