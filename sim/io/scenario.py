"""Lettura dei parametri di scenario da formati diversi."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import TextIO

from sim.model.function import BuiltinRuntime, CustomRuntime, Function, FunctionProfile
from sim.model.infrastructure import Infrastructure
from sim.random.streams import RandomStreams
from sim.utils.validation import (
    finite_time, nonempty_string, nonnegative_integer, positive_integer, positive_number,
)


@dataclass(frozen=True)
class _NodeSpec:
    name: str
    area: str
    machine_tag: str | None


@dataclass(frozen=True)
class _RingSpec:
    id: str
    arch: str
    memory_mb: int
    cpus: float
    replicas: int
    nodes: tuple[_NodeSpec, ...]


@dataclass(frozen=True, init=False)
class Scenario:
    """Configurazione della simulazione e registro delle funzioni."""

    functions: tuple[Function, ...]
    seed: int = 12345
    function_registry: Mapping[str, Function] = field(init=False, repr=False, compare=False)
    _rings: tuple[_RingSpec, ...]
    container_idle_timeout: float
    container_cleanup_interval: float

    def __init__(
        self, functions: Iterable[Function | Mapping[str, object]],
        seed: int = 12345,
        *, infrastructure: Mapping[str, object] | None = None,
        containers: Mapping[str, object] | None = None,
    ) -> None:
        object.__setattr__(self, 'seed', nonnegative_integer(seed, 'seed'))
        if containers is None:
            containers = {}
        if not isinstance(containers, Mapping) or containers.keys() - {'idle_timeout', 'cleanup_interval', 'startup_times'}:
            raise ValueError('containers accetta solo idle_timeout, cleanup_interval e startup_times')
        idle_timeout = finite_time(containers.get('idle_timeout', 600), 'containers.idle_timeout')
        cleanup_interval = finite_time(containers.get('cleanup_interval', 30), 'containers.cleanup_interval')
        if idle_timeout < 0 or cleanup_interval <= 0:
            raise ValueError('idle_timeout deve essere non negativo e cleanup_interval positivo')
        object.__setattr__(self, 'container_idle_timeout', idle_timeout)
        object.__setattr__(self, 'container_cleanup_interval', cleanup_interval)
        startup_times = containers.get('startup_times', {})
        if not isinstance(startup_times, Mapping):
            raise ValueError('containers.startup_times deve associare immagini a durate')
        image_times = {}
        for image, duration in startup_times.items():
            image = nonempty_string(image, 'containers.startup_times (immagine)')
            duration = finite_time(duration, f'containers.startup_times.{image}')
            if duration < 0:
                raise ValueError(f'Il tempo di avvio per {image} deve essere non negativo')
            image_times[image] = duration

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
            runtime = function.runtime
            if isinstance(runtime, (BuiltinRuntime, CustomRuntime)) and runtime.image in image_times:
                runtime = replace(runtime, startup_time=image_times[runtime.image])
            registry[function.name] = replace(
                function, runtime=runtime, sampler=profile.build_sampler(function.name, streams),
            )
        if not registry:
            raise ValueError('Lo scenario richiede almeno una funzione')
        object.__setattr__(self, 'functions', tuple(registry.values()))
        object.__setattr__(self, 'function_registry', MappingProxyType(registry))
        object.__setattr__(self, '_rings', self._load_rings(infrastructure))

    def build_infrastructure(self) -> Infrastructure:
        """Crea ring e nodi nuovi per la replica dallo scenario."""
        infrastructure = Infrastructure()
        for ring in self._rings:
            infrastructure.create_ring(
                ring.id, arch=ring.arch, memory=ring.memory_mb, cpus=ring.cpus, replicas=ring.replicas,
            )
            for node in ring.nodes:
                infrastructure.create_node(
                    ring.id, node.name, node.area, node.machine_tag,
                    idle_timeout=self.container_idle_timeout,
                )
        return infrastructure

    def _load_rings(self, infrastructure: object) -> tuple[_RingSpec, ...]:
        if infrastructure is None:
            return ()
        if not isinstance(infrastructure, Mapping) or infrastructure.keys() != {'rings'}:
            raise ValueError('infrastructure deve contenere rings')
        definitions = infrastructure['rings']
        if not isinstance(definitions, list):
            raise ValueError('infrastructure.rings deve essere una lista')

        rings = {}
        required = {'id', 'arch', 'memory_mb', 'cpus', 'nodes'}
        for definition in definitions:
            if (not isinstance(definition, Mapping) or required - definition.keys()
                    or definition.keys() - required - {'replicas'}):
                raise ValueError('Ogni ring richiede id, arch, memory_mb, cpus e nodes; replicas è opzionale')
            ring_id = nonempty_string(definition['id'], 'ring.id')
            if ring_id in rings:
                raise ValueError(f'Ring già definito: {ring_id}')
            node_definitions = definition['nodes']
            if not isinstance(node_definitions, list):
                raise ValueError(f'I nodi del ring {ring_id} devono essere una lista')
            nodes = {}
            for node in node_definitions:
                if (not isinstance(node, Mapping) or {'name', 'area'} - node.keys()
                        or node.keys() - {'name', 'area', 'machine_tag'}):
                    raise ValueError('Ogni nodo richiede name e area; machine_tag è opzionale')
                name = nonempty_string(node['name'], 'node.name')
                if name in nodes:
                    raise ValueError(f'Nodo già definito nel ring {ring_id}: {name}')
                tag = nonempty_string(node['machine_tag'], 'node.machine_tag') if 'machine_tag' in node else None
                nodes[name] = _NodeSpec(name, nonempty_string(node['area'], 'node.area'), tag)
            rings[ring_id] = _RingSpec(
                ring_id, nonempty_string(definition['arch'], 'ring.arch'),
                positive_integer(definition['memory_mb'], 'ring.memory_mb'),
                positive_number(finite_time(definition['cpus'], 'ring.cpus'), 'ring.cpus'),
                positive_integer(definition.get('replicas', 128), 'ring.replicas'),
                tuple(nodes.values()),
            )
        return tuple(rings.values())

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
        if unknown := data.keys() - {'functions', 'function', 'seed', 'infrastructure', 'containers'}:
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
                    'tar_function_code', 'arrival_distribution', 'service_distribution', 'service_scv',
                    'code_load_time'}
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
                code_load_time=data.get('code_load_time', 0),
            ),
            **metadata,
        )
