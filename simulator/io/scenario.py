"""Lettura dei parametri di scenario da formati diversi."""

import json
import tomllib
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

from simulator.model.execution.function import Function, FunctionProfile
from simulator.model.execution.runtime import BuiltinRuntime, CustomRuntime, Runtime
from simulator.model.scheduling.load_balancer import LoadBalancerMode
from simulator.random.streams import RandomStreams
from simulator.utils.validation import nonempty_string, nonnegative_integer, positive_integer, positive_number

from .execution import _ContainersConfig, _InvocationConfig, _SchedulerConfig
from .functions import _FunctionConfig
from .scaler import ScalerConfig
from .topology import _InfrastructureConfig, _RegionConfig


@dataclass(frozen=True, init=False)
class Scenario:
    """Configurazione della simulazione e registro delle funzioni."""

    function_registry: Mapping[str, Function] = field(init=False, repr=False)
    scheduler: _SchedulerConfig
    containers: _ContainersConfig
    regions: Mapping[str, _RegionConfig]
    primary_region: str
    invocation: _InvocationConfig
    load_balancer: LoadBalancerMode
    load_balancer_replicas: int | None
    load_balancer_resource_cache: bool
    load_balancer_refresh_interval_s: float
    scaler: ScalerConfig
    seed: int = 12345

    def __init__(
        self, functions: Iterable[Function | Mapping[str, object]],
        seed: int = 12345,
        *, infrastructure: Mapping[str, object] | None = None,
        containers: Mapping[str, object] | None = None,
        scheduler: Mapping[str, object] | None = None,
        load_balancer: LoadBalancerMode | str = LoadBalancerMode.ARCHITECTURE_UNAWARE,
        load_balancer_replicas: int | None = None,
        load_balancer_resource_cache: bool = True,
        load_balancer_refresh_interval_s: float = 30.0,
        scaler: ScalerConfig | Mapping[str, object] | str | None = None,
        invocation: Mapping[str, object] | None = None,
        regions: Mapping[str, object] | None = None,
        primary_region: str | None = None,
    ) -> None:
        """Valida le sezioni dello scenario e registra le funzioni."""

        # Il seed e i profili dei container servono alla registrazione delle funzioni.
        seed = nonnegative_integer(seed, 'seed')
        object.__setattr__(self, 'seed', seed)
        try:
            mode = LoadBalancerMode(load_balancer)
        except ValueError as error:
            modes = ', '.join(LoadBalancerMode)
            raise ValueError(f'load_balancer deve essere uno tra {modes}') from error
        object.__setattr__(self, 'load_balancer', mode)
        replicas = (
            positive_integer(load_balancer_replicas, 'load_balancer_replicas')
            if load_balancer_replicas is not None
            else None
        )
        object.__setattr__(self, 'load_balancer_replicas', replicas)
        if not isinstance(load_balancer_resource_cache, bool):
            raise ValueError('load_balancer_resource_cache deve essere un booleano')
        object.__setattr__(self, 'load_balancer_resource_cache', load_balancer_resource_cache)
        refresh_interval = positive_number(load_balancer_refresh_interval_s, 'load_balancer_refresh_interval_s')
        object.__setattr__(self, 'load_balancer_refresh_interval_s', refresh_interval)
        scaler_config = ScalerConfig.parse(scaler)
        object.__setattr__(self, 'scaler', scaler_config)
        scheduler_config = _SchedulerConfig.parse(scheduler)
        object.__setattr__(self, 'scheduler', scheduler_config)
        containers_config = _ContainersConfig.parse(containers)
        object.__setattr__(self, 'containers', containers_config)
        self._configure_regions(regions, primary_region, infrastructure, _InvocationConfig.parse(invocation))

        # Espone il registro in sola lettura e conserva la topologia dichiarata.
        registry = self._register_functions(functions)
        object.__setattr__(self, 'function_registry', MappingProxyType(registry))

    def _configure_regions(
        self, regions: Mapping[str, object] | None, primary_region: str | None,
        infrastructure: Mapping[str, object] | None, invocation: _InvocationConfig,
    ) -> None:
        """Valida la topologia regionale e registra la configurazione di ingresso."""
        if primary_region is not None:
            primary_region = nonempty_string(primary_region, 'primary_region')
        configured, primary = self._parse_regions_map(regions, primary_region, infrastructure)

        # Verifica la regione primaria e la regione di ingresso delle invocazioni.
        self._validate_entry_region(configured, primary, invocation)

        # Le latenze per origine e le destinazioni remote devono indicare altre regioni censite.
        self._validate_region_references(configured)

        # Espone la configurazione regionale in sola lettura.
        object.__setattr__(self, 'primary_region', primary)
        object.__setattr__(self, 'regions', MappingProxyType(configured))
        object.__setattr__(self, 'invocation', invocation)

    @staticmethod
    def _parse_regions_map(
        regions: Mapping[str, object] | None,
        primary_region: str | None,
        infrastructure: Mapping[str, object] | None,
    ) -> tuple[dict[str, _RegionConfig], str]:
        """Costruisce la mappa delle regioni e determina la regione primaria."""
        if regions is None:
            primary = primary_region or 'default'
            infrastructure_config = _InfrastructureConfig.parse(infrastructure)
            return {primary: _RegionConfig(infrastructure=infrastructure_config)}, primary

        if infrastructure is not None:
            raise ValueError('Con regions, dichiarare infrastructure dentro ogni regione')
        if not isinstance(regions, Mapping) or not regions:
            raise ValueError('regions deve essere una tabella di regioni non vuota')

        configured = {
            nonempty_string(name, 'regions (nome)'): _RegionConfig.parse(nonempty_string(name, 'regions (nome)'), definition)
            for name, definition in regions.items()
        }
        return configured, primary_region or next(iter(configured))

    @staticmethod
    def _validate_entry_region(
        configured: Mapping[str, _RegionConfig], primary: str, invocation: _InvocationConfig,
    ) -> None:
        """Verifica che la regione primaria e la regione di ingresso siano valide."""
        if primary not in configured:
            raise ValueError(f'primary_region sconosciuta: {primary}')

        entry = invocation.entry_region or primary
        if entry not in configured:
            raise ValueError(f'invocation.entry_region sconosciuta: {entry}')
        if configured[entry].infrastructure is None and invocation.entry_node:
            raise ValueError(f'La regione {entry} accetta invocazioni a livello di regione')

    @classmethod
    def _validate_region_references(cls, configured: Mapping[str, _RegionConfig]) -> None:
        """Verifica che latenze sorgente e destinazioni remote puntino ad altre regioni censite."""
        for name, region in configured.items():
            for source in region.offload_latency_by_source_s:
                if source not in configured or source == name:
                    raise ValueError(
                        f"regions.{name}.offload_latency_by_source_s richiede un'altra regione censita: {source}"
                    )
            if region.infrastructure is not None:
                cls._validate_node_remote_regions(name, region.infrastructure, configured)

    @staticmethod
    def _validate_node_remote_regions(
        region_name: str, infrastructure: _InfrastructureConfig, configured: Mapping[str, _RegionConfig],
    ) -> None:
        """Verifica le destinazioni remote dichiarate sui nodi di una regione."""
        for ring in infrastructure.rings:
            for node in ring.nodes:
                remote = node.remote_region
                if remote is not None and (remote not in configured or remote == region_name):
                    raise ValueError(f"node.remote_region deve indicare un'altra regione censita: {remote}")

    def _register_functions(
        self, definitions: Iterable[Function | Mapping[str, object]],
    ) -> OrderedDict[str, Function]:
        """Costruisce l'anagrafica e assegna i flussi di arrivo e servizio."""
        registry: OrderedDict[str, Function] = OrderedDict()
        streams = RandomStreams(self.seed)

        for index, definition in enumerate(definitions):
            function = self._parse_function_definition(index, definition)
            profile = function.profile
            if not isinstance(profile, FunctionProfile):
                raise ValueError(f'Profilo di simulazione mancante o non valido per {function.name}')
            if function.name in registry:
                raise ValueError(f'Funzione già registrata: {function.name}')

            runtime = self._resolve_function_runtime(definition, function)
            sampler = profile.build_sampler(function.name, streams)
            registry[function.name] = replace(function, runtime=runtime, sampler=sampler)

        if not registry:
            raise ValueError('Lo scenario richiede almeno una funzione')
        return registry

    def _parse_function_definition(self, index: int, definition: Function | Mapping[str, object]) -> Function:
        """Converte una definizione di funzione in oggetto Function."""
        if isinstance(definition, Function):
            return definition
        try:
            function_config = _FunctionConfig.parse(definition, startup_times=self.containers.startup_times)
            return function_config.build_function()
        except ValueError as error:
            raise ValueError(f'functions[{index}]: {error}') from error

    def _resolve_function_runtime(
        self, definition: Function | Mapping[str, object], function: Function,
    ) -> Runtime:
        """Applica i tempi di startup dello scenario anche alle funzioni già istanziate."""
        runtime = function.runtime
        if not isinstance(definition, Function):
            return runtime
        name = runtime.name if isinstance(runtime, BuiltinRuntime) else 'custom'
        image = runtime.image if isinstance(runtime, CustomRuntime) else None
        return Runtime.get_runtime(
            name=name,
            custom_image=image,
            startup_time=runtime.startup_time,
            startup_times=self.containers.startup_times,
        )

    @classmethod
    def load(cls, path: str | Path) -> 'Scenario':
        """Carica e valida uno scenario dal formato indicato dall'estensione."""
        # Seleziona il lettore in base all'estensione del file.
        path = Path(path)
        readers = {
            '.yaml': cls._read_yaml,
            '.yml': cls._read_yaml,
            '.toml': cls._read_toml,
            '.json': cls._read_json,
        }
        reader = readers.get(path.suffix.lower())
        if reader is None:
            raise ValueError(f'Formato scenario non supportato: {path.suffix or "senza estensione"}')
        with path.open(encoding='utf-8') as source:
            data = reader(source)
        # Controlla le sezioni ammesse prima di validarne il contenuto.
        if not isinstance(data, dict):
            raise ValueError('Lo scenario deve essere una tabella')
        allowed = {'functions', 'function', 'seed', 'infrastructure', 'containers', 'scheduler',
                   'load_balancer', 'load_balancer_replicas', 'load_balancer_resource_cache',
                   'load_balancer_refresh_interval_s', 'scaler', 'invocation', 'regions', 'primary_region'}
        if unknown := data.keys() - allowed:
            raise ValueError(f'Parametri scenario sconosciuti: {sorted(map(str, unknown))}')

        # Normalizza la forma a funzione singola nella lista usata dal costruttore.
        if 'function' in data and 'functions' in data:
            raise ValueError('Usare un solo campo tra function e functions')
        definitions = [data['function']] if 'function' in data else data.get('functions')
        if not isinstance(definitions, list):
            raise ValueError('functions deve essere una lista di definizioni')

        return cls(
            functions=definitions,
            seed=data.get('seed', 12345),
            infrastructure=data.get('infrastructure'),
            containers=data.get('containers'),
            scheduler=data.get('scheduler'),
            load_balancer=data.get('load_balancer', LoadBalancerMode.ARCHITECTURE_UNAWARE),
            load_balancer_replicas=data.get('load_balancer_replicas'),
            load_balancer_resource_cache=data.get('load_balancer_resource_cache', True),
            load_balancer_refresh_interval_s=data.get('load_balancer_refresh_interval_s', 30.0),
            scaler=data.get('scaler'),
            invocation=data.get('invocation'),
            regions=data.get('regions'),
            primary_region=data.get('primary_region'),
        )

    @staticmethod
    def _read_yaml(source: TextIO) -> object:
        """Legge la configurazione YAML."""
        import yaml

        try:
            return yaml.safe_load(source)
        except yaml.YAMLError as error:
            raise ValueError(f'YAML non valido: {error}') from error

    @staticmethod
    def _read_toml(source: TextIO) -> object:
        """Legge la configurazione TOML."""
        return tomllib.loads(source.read())

    @staticmethod
    def _read_json(source: TextIO) -> object:
        """Legge la configurazione JSON."""
        return json.load(source)

    def __getstate__(self) -> dict:
        return {
            'function_registry': dict(self.function_registry),
            'scheduler': self.scheduler,
            'containers': self.containers,
            'regions': dict(self.regions),
            'primary_region': self.primary_region,
            'invocation': self.invocation,
            'load_balancer': self.load_balancer,
            'load_balancer_replicas': self.load_balancer_replicas,
            'load_balancer_resource_cache': self.load_balancer_resource_cache,
            'load_balancer_refresh_interval_s': self.load_balancer_refresh_interval_s,
            'scaler': self.scaler,
            'seed': self.seed,
        }

    def __setstate__(self, state: dict) -> None:
        for key, value in state.items():
            if key in ('function_registry', 'regions'):
                value = MappingProxyType(value)
            object.__setattr__(self, key, value)
