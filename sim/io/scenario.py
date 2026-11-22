"""Lettura dei parametri di scenario da formati diversi."""

import json
import tomllib
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

from sim.model.function import Function, FunctionProfile
from sim.model.load_balancer import LoadBalancerMode
from sim.model.runtime import BuiltinRuntime, CustomRuntime, Runtime
from sim.random.streams import RandomStreams
from sim.utils.validation import nonempty_string, nonnegative_integer
from ._scenario_config import (
    _ContainersConfig, _FunctionConfig, _InfrastructureConfig, _InvocationConfig, _RegionConfig, _SchedulerConfig,
)


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
    seed: int = 12345

    def __init__(
        self, functions: Iterable[Function | Mapping[str, object]],
        seed: int = 12345,
        *, infrastructure: Mapping[str, object] | None = None,
        containers: Mapping[str, object] | None = None,
        scheduler: Mapping[str, object] | None = None,
        load_balancer: LoadBalancerMode | str = LoadBalancerMode.RANDOM,
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

        # Se manca la sezione regions, usa l'infrastruttura come unica regione.
        if primary_region is not None:
            primary_region = nonempty_string(primary_region, 'primary_region')
        if regions is None:
            primary = primary_region or 'default'
            infrastructure_config = _InfrastructureConfig.parse(infrastructure)
            configured = {primary: _RegionConfig(infrastructure=infrastructure_config)}
        else:
            if infrastructure is not None:
                raise ValueError('Con regions, dichiarare infrastructure dentro ogni regione')
            if not isinstance(regions, Mapping) or not regions:
                raise ValueError('regions deve essere una tabella di regioni non vuota')
            configured = {}
            for name, definition in regions.items():
                name = nonempty_string(name, 'regions (nome)')
                configured[name] = _RegionConfig.parse(name, definition)
            primary = primary_region or next(iter(configured))

        # Verifica la regione primaria e la regione di ingresso delle invocazioni.
        if primary not in configured:
            raise ValueError(f'primary_region sconosciuta: {primary}')

        entry = invocation.entry_region or primary
        if entry not in configured:
            raise ValueError(f'invocation.entry_region sconosciuta: {entry}')
        if configured[entry].infrastructure is None and invocation.entry_node:
            raise ValueError(f'La regione {entry} accetta invocazioni a livello di regione')

        # Le latenze per origine e le destinazioni remote devono indicare altre regioni censite.
        for name, region in configured.items():
            for source in region.offload_latency_by_source_s:
                if source not in configured or source == name:
                    raise ValueError(f"regions.{name}.offload_latency_by_source_s richiede un'altra regione censita: {source}")
            if region.infrastructure is None:
                continue
            for ring in region.infrastructure.rings:
                for node in ring.nodes:
                    remote = node.remote_region
                    if remote is not None and (remote not in configured or remote == name):
                        raise ValueError(f"node.remote_region deve indicare un'altra regione censita: {remote}")

        # Espone la configurazione regionale in sola lettura.
        object.__setattr__(self, 'primary_region', primary)
        object.__setattr__(self, 'regions', MappingProxyType(configured))
        object.__setattr__(self, 'invocation', invocation)

    def _register_functions(
        self, definitions: Iterable[Function | Mapping[str, object]],
    ) -> OrderedDict[str, Function]:
        """Costruisce l'anagrafica e assegna i flussi di arrivo e servizio."""

        # L'ordine delle funzioni conta anche quando si confrontano due scenari.
        registry: OrderedDict[str, Function] = OrderedDict()

        # Una sola fabbrica per lo scenario assegna flussi distinti a ogni funzione.
        streams = RandomStreams(self.seed)

        # Costruisce le funzioni e le registra con i campionatori.
        for index, definition in enumerate(definitions):

            # Accetta anche modelli già costruiti e identifica le definizioni invalide.
            if isinstance(definition, Function):
                function = definition
            else:
                try:
                    function_config = _FunctionConfig.parse(definition, startup_times=self.containers.startup_times)
                    function = function_config.build_function()
                except ValueError as error:
                    raise ValueError(f'functions[{index}]: {error}') from error

            # Ogni funzione deve avere un profilo e un nome univoco nel registro.
            profile = function.profile
            if not isinstance(profile, FunctionProfile):
                raise ValueError(f'Profilo di simulazione mancante o non valido per {function.name}')
            if function.name in registry:
                raise ValueError(f'Funzione già registrata: {function.name}')

            # Anche i modelli passati direttamente usano il runtime configurato dallo scenario.
            runtime = function.runtime
            if isinstance(definition, Function):
                name = runtime.name if isinstance(runtime, BuiltinRuntime) else 'custom'
                image = runtime.image if isinstance(runtime, CustomRuntime) else None
                runtime = Runtime.get_runtime(
                    name=name,
                    custom_image=image,
                    startup_time=runtime.startup_time,
                    startup_times=self.containers.startup_times,
                )

            # Assegna i flussi di arrivo e servizio della funzione registrata.
            sampler = profile.build_sampler(function.name, streams)
            registry[function.name] = replace(function, runtime=runtime, sampler=sampler)

        # Lo scenario deve poter pianificare almeno un arrivo iniziale.
        if not registry:
            raise ValueError('Lo scenario richiede almeno una funzione')
        return registry

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
                   'load_balancer', 'invocation', 'regions', 'primary_region'}
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
            load_balancer=data.get('load_balancer', LoadBalancerMode.RANDOM),
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
