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

from simulator.model.scheduling.load_balancer import LoadBalancerMode
from simulator.model.topology.region import RegionMode
from simulator.utils.validation import nonempty_string, nonnegative_number, positive_integer, positive_number

from .infrastructure import _InfrastructureConfig
from ..scaler import ScalerConfig


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
    load_balancer_resource_cache: bool | None = None
    load_balancer_refresh_interval_s: float | None = None
    scaler: ScalerConfig | None = None

    @classmethod
    def parse(cls, name: str, data: object) -> '_RegionConfig':
        """Parse della configurazione di una regione."""
        prefix = f'regions.{name}'
        allowed = {
            'mode', 'infrastructure', 'service_time_s', 'offload_latency_s', 'offload_latency_by_source_s',
            'has_load_balancer', 'load_balancer', 'load_balancer_replicas',
            'load_balancer_resource_cache', 'load_balancer_refresh_interval_s', 'scaler',
        }
        if not isinstance(data, Mapping) or data.keys() - allowed:
            raise ValueError(f'Configurazione non valida per {prefix}')

        latency = nonnegative_number(data.get('offload_latency_s', 0), f'{prefix}.offload_latency_s')
        latency_by_source = cls._parse_latency_by_source(data.get('offload_latency_by_source_s', {}), prefix)
        mode = data.get('mode', RegionMode.SIMULATED)

        if mode == RegionMode.SIMULATED:
            return cls._parse_simulated(data, prefix, latency, latency_by_source)
        if mode == RegionMode.ALWAYS_SUCCEEDS:
            return cls._parse_always_succeeds(data, prefix, latency, latency_by_source)
        raise ValueError(f'{prefix}.mode deve essere {RegionMode.SIMULATED.value} o {RegionMode.ALWAYS_SUCCEEDS.value}')

    @staticmethod
    def _parse_latency_by_source(by_source: object, prefix: str) -> Mapping[str, float]:
        """Valida la mappa delle latenze di offloading per regione sorgente."""
        by_source_field = f'{prefix}.offload_latency_by_source_s'
        if not isinstance(by_source, Mapping):
            raise ValueError(f'{by_source_field} deve associare regioni di origine a durate')
        latency_by_source = {}
        for source, duration in by_source.items():
            source_name = nonempty_string(source, by_source_field)
            latency_by_source[source_name] = nonnegative_number(duration, f'{by_source_field}.{source_name}')
        return MappingProxyType(latency_by_source)

    @staticmethod
    def _parse_optional_bool(data: Mapping[str, object], key: str, prefix: str) -> bool | None:
        """Valida un campo booleano opzionale."""
        if key not in data:
            return None
        value = data[key]
        if not isinstance(value, bool):
            raise ValueError(f'{prefix}.{key} deve essere un booleano')
        return value

    @staticmethod
    def _parse_optional_load_balancer(data: Mapping[str, object], prefix: str) -> LoadBalancerMode | None:
        """Valida la modalità del load balancer regionale, se dichiarata."""
        if 'load_balancer' not in data:
            return None
        try:
            return LoadBalancerMode(data['load_balancer'])
        except ValueError as error:
            modes = ', '.join(LoadBalancerMode)
            raise ValueError(f'{prefix}.load_balancer deve essere uno tra {modes}') from error

    @classmethod
    def _parse_simulated(
        cls,
        data: Mapping[str, object],
        prefix: str,
        latency: float,
        latency_by_source: Mapping[str, float],
    ) -> '_RegionConfig':
        """Costruisce la configurazione di una regione in modalità simulata."""
        if 'service_time_s' in data:
            raise ValueError(f'{prefix}.service_time_s richiede mode = "{RegionMode.ALWAYS_SUCCEEDS.value}"')

        has_lb = cls._parse_optional_bool(data, 'has_load_balancer', prefix)
        load_balancer_replicas = (
            positive_integer(data['load_balancer_replicas'], f'{prefix}.load_balancer_replicas')
            if 'load_balancer_replicas' in data
            else None
        )
        load_balancer_refresh_interval_s = (
            positive_number(data['load_balancer_refresh_interval_s'], f'{prefix}.load_balancer_refresh_interval_s')
            if 'load_balancer_refresh_interval_s' in data
            else None
        )
        scaler = ScalerConfig.parse(data['scaler'], prefix=f'{prefix}.scaler') if 'scaler' in data else None

        return cls(
            infrastructure=_InfrastructureConfig.parse(data.get('infrastructure')),
            service_time_s=0.0,
            offload_latency_s=latency,
            offload_latency_by_source_s=latency_by_source,
            has_load_balancer=True if has_lb is None else has_lb,
            load_balancer=cls._parse_optional_load_balancer(data, prefix),
            load_balancer_replicas=load_balancer_replicas,
            load_balancer_resource_cache=cls._parse_optional_bool(data, 'load_balancer_resource_cache', prefix),
            load_balancer_refresh_interval_s=load_balancer_refresh_interval_s,
            scaler=scaler,
        )

    @classmethod
    def _parse_always_succeeds(
        cls,
        data: Mapping[str, object],
        prefix: str,
        latency: float,
        latency_by_source: Mapping[str, float],
    ) -> '_RegionConfig':
        """Costruisce la configurazione di una regione sintetica always_succeeds."""
        simulated_only = (
            'infrastructure', 'has_load_balancer', 'load_balancer', 'load_balancer_replicas',
            'load_balancer_resource_cache', 'load_balancer_refresh_interval_s', 'scaler',
        )
        for field_name in simulated_only:
            if field_name in data:
                raise ValueError(f'{prefix}.{field_name} richiede mode = "{RegionMode.SIMULATED.value}"')

        return cls(
            infrastructure=None,
            service_time_s=nonnegative_number(data.get('service_time_s', 0), f'{prefix}.service_time_s'),
            offload_latency_s=latency,
            offload_latency_by_source_s=latency_by_source,
            has_load_balancer=True,
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
            'load_balancer_resource_cache': self.load_balancer_resource_cache,
            'load_balancer_refresh_interval_s': self.load_balancer_refresh_interval_s,
            'scaler': self.scaler,
        }

    def __setstate__(self, state: dict) -> None:
        for key, value in state.items():
            if key == 'offload_latency_by_source_s':
                value = MappingProxyType(value)
            object.__setattr__(self, key, value)
