"""Lettura dei parametri di scenario da formati diversi."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import TextIO

from sim.model.function import BuiltinRuntime, CustomRuntime, Function, FunctionProfile
from sim.random.streams import RandomStreams
from sim.utils.validation import finite_time, nonempty_string, nonnegative_integer, positive_integer


@dataclass(frozen=True, init=False)
class Scenario:
    """Configurazione della simulazione e registro delle funzioni."""

    functions: tuple[Function, ...]
    seed: int = 12345
    function_registry: Mapping[str, Function] = field(init=False, repr=False, compare=False)

    def __init__(
        self, functions: Iterable[Function | Mapping[str, object]],
        seed: int = 12345,
    ) -> None:
        object.__setattr__(self, 'seed', nonnegative_integer(seed, 'seed'))
        registry: dict[str, Function] = {}
        streams = RandomStreams(self.seed)
        for index, definition in enumerate(functions):
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
            registry[function.name] = replace(function, sampler=profile.build_sampler(function.name, streams))
        if not registry:
            raise ValueError('Lo scenario richiede almeno una funzione')
        object.__setattr__(self, 'functions', tuple(registry.values()))
        object.__setattr__(self, 'function_registry', MappingProxyType(registry))

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
        if unknown := data.keys() - {'functions', 'function', 'seed'}:
            raise ValueError(f'Parametri scenario sconosciuti: {sorted(map(str, unknown))}')

        if 'function' in data and 'functions' in data:
            raise ValueError('Usare un solo campo tra function e functions')
        definitions = [data['function']] if 'function' in data else data.get('functions')
        if not isinstance(definitions, list):
            raise ValueError('functions deve essere una lista di definizioni')

        return cls(
            functions=definitions,
            seed=data.get('seed', 12345),
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
        required = {'name', 'runtime', 'memory_mb', 'cpu_demand', 'arrival_rate', 'service_rate', 'first_arrival_time'}
        optional = {'custom_image', 'max_concurrency', 'supported_archs', 'tag_pattern',
                    'tar_function_code', 'arrival_distribution', 'service_distribution', 'service_scv'}
        if not isinstance(data, Mapping):
            raise ValueError('La definizione della funzione deve essere una tabella')
        if missing := required - data.keys():
            raise ValueError(f'Campi obbligatori mancanti: {sorted(missing)}')
        if unknown := data.keys() - required - optional:
            raise ValueError(f'Campi sconosciuti: {sorted(map(str, unknown))}')

        runtime_name = nonempty_string(data['runtime'], 'runtime')
        if runtime_name == 'custom':
            runtime = CustomRuntime(nonempty_string(data.get('custom_image'), 'custom_image'))
        else:
            if 'custom_image' in data:
                raise ValueError('custom_image è ammesso solo per il runtime custom')
            runtime = BuiltinRuntime(runtime_name)

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

        return Function(
            name=nonempty_string(data['name'], 'name'),
            runtime=runtime,
            memory_mb=positive_integer(data['memory_mb'], 'memory_mb'),
            cpu_demand=finite_time(data['cpu_demand'], 'cpu_demand'),
            max_concurrency=positive_integer(data.get('max_concurrency', 1), 'max_concurrency'),
            supported_archs=supported_archs,
            profile=FunctionProfile(
                arrival_rate=data['arrival_rate'],
                service_rate=data['service_rate'],
                first_arrival_time=data['first_arrival_time'],
                arrival_distribution=data.get('arrival_distribution', 'poisson'),
                service_distribution=data.get('service_distribution', 'gamma'),
                service_scv=data.get('service_scv'),
            ),
            **metadata,
        )
