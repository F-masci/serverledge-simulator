"""Lettura dei parametri di scenario da formati diversi."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import TextIO

from sim.model.function import Function, FunctionProfile
from sim.model.infrastructure import Infrastructure
from sim.random.streams import RandomStreams
from sim.utils.validation import nonnegative_integer

from ._scenario_config import _ContainersConfig, _FunctionConfig, _InfrastructureConfig, _SchedulerConfig


@dataclass(frozen=True, init=False)
class Scenario:
    """Configurazione della simulazione e registro delle funzioni."""

    functions: tuple[Function, ...]
    seed: int = 12345
    function_registry: Mapping[str, Function] = field(init=False, repr=False, compare=False)
    _scheduler: _SchedulerConfig
    _containers: _ContainersConfig
    _infrastructure: _InfrastructureConfig

    def __init__(
        self, functions: Iterable[Function | Mapping[str, object]],
        seed: int = 12345,
        *, infrastructure: Mapping[str, object] | None = None,
        containers: Mapping[str, object] | None = None,
        scheduler: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, 'seed', nonnegative_integer(seed, 'seed'))
        object.__setattr__(self, '_scheduler', _SchedulerConfig.parse(scheduler))
        object.__setattr__(self, '_containers', _ContainersConfig.parse(containers))
        registry = self._register_functions(functions)
        object.__setattr__(self, 'functions', tuple(registry.values()))
        object.__setattr__(self, 'function_registry', MappingProxyType(registry))
        object.__setattr__(self, '_infrastructure', _InfrastructureConfig.parse(infrastructure))

    @property
    def queue_capacity(self) -> int:
        return self._scheduler.queue_capacity

    @property
    def container_idle_timeout(self) -> float:
        return self._containers.idle_timeout

    @property
    def container_cleanup_interval(self) -> float:
        return self._containers.cleanup_interval

    def _register_functions(
        self, definitions: Iterable[Function | Mapping[str, object]],
    ) -> dict[str, Function]:
        """Costruisce l'anagrafica e assegna i flussi di arrivo e servizio."""
        registry: dict[str, Function] = {}
        # Una sola fabbrica per lo scenario assegna flussi distinti a ogni funzione.
        streams = RandomStreams(self.seed)
        for index, definition in enumerate(definitions):
            if isinstance(definition, Function):
                function = definition
            else:
                try:
                    function = self._load_function(definition)
                except ValueError as error:
                    raise ValueError(f'functions[{index}]: {error}') from error
            profile = function.profile
            if not isinstance(profile, FunctionProfile):
                raise ValueError(f'Profilo di simulazione mancante o non valido per {function.name}')
            if function.name in registry:
                raise ValueError(f'Funzione già registrata: {function.name}')
            runtime = self._containers.configure_runtime(function.runtime)
            registry[function.name] = replace(
                function, runtime=runtime, sampler=profile.build_sampler(function.name, streams),
            )
        if not registry:
            raise ValueError('Lo scenario richiede almeno una funzione')
        return registry

    def build_infrastructure(self) -> Infrastructure:
        """Crea ring e nodi nuovi per la replica dallo scenario."""
        infrastructure = Infrastructure()
        for ring in self._infrastructure.rings:
            infrastructure.create_ring(
                ring.id, arch=ring.arch, memory=ring.memory_mb, cpus=ring.cpus, replicas=ring.replicas,
            )
            for node in ring.nodes:
                infrastructure.create_node(
                    ring.id, node.name, node.area, node.machine_tag,
                    idle_timeout=self.container_idle_timeout, queue_capacity=self.queue_capacity,
                )
        return infrastructure

    def get_function(self, name: str) -> Function | None:
        """Cerca una funzione registrata per nome."""
        return self.function_registry.get(name)

    @classmethod
    def load(cls, path: str | Path) -> 'Scenario':
        """Carica e valida uno scenario dal formato indicato dall'estensione."""
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
        if not isinstance(data, dict):
            raise ValueError('Lo scenario deve essere una tabella')
        if unknown := data.keys() - {'functions', 'function', 'seed', 'infrastructure', 'containers', 'scheduler'}:
            raise ValueError(f'Parametri scenario sconosciuti: {sorted(map(str, unknown))}')

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
        )

    @staticmethod
    def _read_yaml(source: TextIO) -> object:
        import yaml

        try:
            return yaml.safe_load(source)
        except yaml.YAMLError as error:
            raise ValueError(f'YAML non valido: {error}') from error

    @staticmethod
    def _read_toml(source: TextIO) -> object:
        return tomllib.loads(source.read())

    @staticmethod
    def _read_json(source: TextIO) -> object:
        return json.load(source)

    def _load_function(self, data: object) -> Function:
        """Converte una definizione esterna prima della registrazione."""
        return _FunctionConfig.parse(data).build()
