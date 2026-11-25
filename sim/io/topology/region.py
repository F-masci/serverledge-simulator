"""Configurazione di una regione geografica nello scenario.

Esempio TOML nello scenario — regione simulata (mode = "simulated", default):

    [regions.edge]
    mode = "simulated"                    # opzionale (default: "simulated")
    has_load_balancer = true              # opzionale (default: true)
    load_balancer = "RoundRobin"          # opzionale (default: load_balancer globale)
    load_balancer_replicas = 128          # opzionale (default: globale o 128)
    offload_latency_s = 0.0               # opzionale (default: 0.0 s, RTT di default)
    # infrastructure = { rings = [...] }  # opzionale (default: nessun ring)
    # scaler = { mode = "PPO", ... }      # opzionale (default: scaler globale)

    [regions.edge.offload_latency_by_source_s]
    regional = 0.25                       # opzionale: regione sorgente -> RTT in secondi

Esempio TOML — regione sintetica (mode = "always_succeeds"):

    [regions.cloud]
    mode = "always_succeeds"              # obbligatorio per modalità sintetica
    service_time_s = 0.375                # opzionale (default: 0.0 s; solo in always_succeeds)
    offload_latency_s = 0.5               # opzionale (default: 0.0 s)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from sim.model.load_balancer import LoadBalancerMode
from sim.model.region import RegionMode
from sim.utils.validation import nonempty_string, nonnegative_number, positive_integer
from ..scaler import ScalerConfig
from .infrastructure import _InfrastructureConfig


@dataclass(frozen=True)
class _RegionConfig:
    """Topologia, modalità di esecuzione, load balancer, scaler e latenze di una regione."""

    infrastructure: _InfrastructureConfig | None
    service_time_s: float = 0.0
    offload_latency_s: float = 0.0
    offload_latency_by_source_s: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    has_load_balancer: bool = True
    load_balancer: LoadBalancerMode | None = None
    load_balancer_replicas: int | None = None
    scaler: ScalerConfig | None = None

    @classmethod
    def parse(cls, name: str, data: object) -> '_RegionConfig':
        """Parse della configurazione di una regione."""
        prefix = f'regions.{name}'
        allowed = {
            'mode', 'infrastructure', 'service_time_s', 'offload_latency_s', 'offload_latency_by_source_s',
            'has_load_balancer', 'load_balancer', 'load_balancer_replicas', 'scaler',
        }
        if not isinstance(data, Mapping) or data.keys() - allowed:
            raise ValueError(f'Configurazione non valida per {prefix}')

        latency = nonnegative_number(data.get('offload_latency_s', 0), f'{prefix}.offload_latency_s')
        by_source_field = f'{prefix}.offload_latency_by_source_s'
        by_source = data.get('offload_latency_by_source_s', {})
        if not isinstance(by_source, Mapping):
            raise ValueError(f'{by_source_field} deve associare regioni di origine a durate')
        latency_by_source = {}
        for source, duration in by_source.items():
            source = nonempty_string(source, by_source_field)
            latency_by_source[source] = nonnegative_number(duration, f'{by_source_field}.{source}')

        mode = data.get('mode', RegionMode.SIMULATED)
        load_balancer: LoadBalancerMode | None = None
        load_balancer_replicas: int | None = None
        scaler: ScalerConfig | None = None
        if mode == RegionMode.SIMULATED:
            if 'service_time_s' in data:
                raise ValueError(f'{prefix}.service_time_s richiede mode = "{RegionMode.ALWAYS_SUCCEEDS.value}"')
            infrastructure = _InfrastructureConfig.parse(data.get('infrastructure'))
            service_time = 0.0
            has_lb = data.get('has_load_balancer', True)
            if not isinstance(has_lb, bool):
                raise ValueError(f'{prefix}.has_load_balancer deve essere un booleano')
            has_load_balancer = has_lb
            if 'load_balancer' in data:
                try:
                    load_balancer = LoadBalancerMode(data['load_balancer'])
                except ValueError as error:
                    modes = ', '.join(LoadBalancerMode)
                    raise ValueError(f'{prefix}.load_balancer deve essere uno tra {modes}') from error
            if 'load_balancer_replicas' in data:
                load_balancer_replicas = positive_integer(
                    data['load_balancer_replicas'], f'{prefix}.load_balancer_replicas',
                )
            if 'scaler' in data:
                scaler = ScalerConfig.parse(data['scaler'], prefix=f'{prefix}.scaler')
        elif mode == RegionMode.ALWAYS_SUCCEEDS:
            for field_name in ('infrastructure', 'has_load_balancer', 'load_balancer', 'load_balancer_replicas', 'scaler'):
                if field_name in data:
                    raise ValueError(f'{prefix}.{field_name} richiede mode = "{RegionMode.SIMULATED.value}"')
            infrastructure = None
            service_time = nonnegative_number(data.get('service_time_s', 0), f'{prefix}.service_time_s')
            has_load_balancer = True
        else:
            raise ValueError(f'{prefix}.mode deve essere {RegionMode.SIMULATED.value} o {RegionMode.ALWAYS_SUCCEEDS.value}')

        return cls(
            infrastructure=infrastructure,
            service_time_s=service_time,
            offload_latency_s=latency,
            offload_latency_by_source_s=MappingProxyType(latency_by_source),
            has_load_balancer=has_load_balancer,
            load_balancer=load_balancer,
            load_balancer_replicas=load_balancer_replicas,
            scaler=scaler,
        )

    def __getstate__(self) -> dict:
        return {
            'infrastructure': self.infrastructure,
            'service_time_s': self.service_time_s,
            'offload_latency_s': self.offload_latency_s,
            'offload_latency_by_source_s': dict(self.offload_latency_by_source_s),
            'has_load_balancer': self.has_load_balancer,
            'load_balancer': self.load_balancer,
            'load_balancer_replicas': self.load_balancer_replicas,
            'scaler': self.scaler,
        }

    def __setstate__(self, state: dict) -> None:
        for key, value in state.items():
            if key == 'offload_latency_by_source_s':
                value = MappingProxyType(value)
            object.__setattr__(self, key, value)
