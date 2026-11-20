"""Lettura dei parametri di scenario da formati diversi."""

from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import TextIO

from sim.model.function import BuiltinRuntime, CustomRuntime, Function, FunctionProfile, Runtime
from sim.random.streams import RandomStreams
from sim.utils.validation import nonnegative_integer

from ._scenario_config import (
    _ContainersConfig, _FunctionConfig, _InfrastructureConfig, _SchedulerConfig,
)


@dataclass(frozen=True, init=False)
class Scenario:
    """Configurazione della simulazione e registro delle funzioni."""

    function_registry: Mapping[str, Function] = field(init=False, repr=False)
    scheduler: _SchedulerConfig
    containers: _ContainersConfig
    infrastructure: _InfrastructureConfig
    seed: int = 12345

    def __init__(
        self, functions: Iterable[Function | Mapping[str, object]],
        seed: int = 12345,
        *, infrastructure: Mapping[str, object] | None = None,
        containers: Mapping[str, object] | None = None,
        scheduler: Mapping[str, object] | None = None,
    ) -> None:
        """Valida le sezioni dello scenario e registra le funzioni."""

        # Il seed e i profili dei container servono alla registrazione delle funzioni.
        seed = nonnegative_integer(seed, 'seed')
        object.__setattr__(self, 'seed', seed)
        scheduler_config = _SchedulerConfig.parse(scheduler)
        object.__setattr__(self, 'scheduler', scheduler_config)
        containers_config = _ContainersConfig.parse(containers)
        object.__setattr__(self, 'containers', containers_config)
        infrastructure_config = _InfrastructureConfig.parse(infrastructure)
        object.__setattr__(self, 'infrastructure', infrastructure_config)

        # Espone il registro in sola lettura e conserva la topologia dichiarata.
        registry = self._register_functions(functions)
        object.__setattr__(self, 'function_registry', MappingProxyType(registry))

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
                    fConf = _FunctionConfig.parse(definition, startup_times=self.containers.startup_times)
                    function = fConf.build_function()
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
        if unknown := data.keys() - {'functions', 'function', 'seed', 'infrastructure', 'containers', 'scheduler'}:
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
