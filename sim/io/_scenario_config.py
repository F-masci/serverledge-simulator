"""Sezioni interne della configurazione di uno scenario."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

from sim.model.function import BuiltinRuntime, CustomRuntime, Function, FunctionProfile, Runtime
from sim.utils.validation import finite_time, integer, nonempty_string, positive_integer, positive_number


@dataclass(frozen=True)
class _SchedulerConfig:
    """Parametri della politica di ammissione locale, condivisi dai nodi."""

    queue_capacity: int

    @classmethod
    def parse(cls, scheduler: object) -> '_SchedulerConfig':
        if scheduler is None:
            scheduler = {}
        if not isinstance(scheduler, Mapping) or scheduler.keys() - {'queue'}:
            raise ValueError('scheduler accetta solo queue')
        queue = scheduler.get('queue', {})
        if not isinstance(queue, Mapping) or queue.keys() - {'capacity'}:
            raise ValueError('scheduler.queue accetta solo capacity')
        return cls(queue_capacity=integer(queue.get('capacity', 0), 'scheduler.queue.capacity'))


@dataclass(frozen=True)
class _ContainersConfig:
    """Politica per i container idle e tempi di avvio per immagine Docker."""

    idle_timeout: float
    cleanup_interval: float
    startup_times: Mapping[str, float]

    @classmethod
    def parse(cls, containers: object) -> '_ContainersConfig':
        if containers is None:
            containers = {}
        if not isinstance(containers, Mapping) or containers.keys() - {'idle_timeout', 'cleanup_interval', 'startup_times'}:
            raise ValueError('containers accetta solo idle_timeout, cleanup_interval e startup_times')
        idle_timeout = finite_time(containers.get('idle_timeout', 600), 'containers.idle_timeout')
        cleanup_interval = finite_time(containers.get('cleanup_interval', 30), 'containers.cleanup_interval')
        if idle_timeout < 0 or cleanup_interval <= 0:
            raise ValueError('idle_timeout deve essere non negativo e cleanup_interval positivo')
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
        return cls(idle_timeout, cleanup_interval, MappingProxyType(image_times))

    def configure_runtime(self, runtime: Runtime) -> Runtime:
        """Applica il profilo condiviso dall'immagine, se definito nello scenario."""
        if isinstance(runtime, (BuiltinRuntime, CustomRuntime)) and runtime.image in self.startup_times:
            return replace(runtime, startup_time=self.startup_times[runtime.image])
        return runtime


@dataclass(frozen=True)
class _NodeConfig:
    """Identità del nodo; architettura e capacità sono definite dal ring."""

    name: str
    area: str
    machine_tag: str | None

    @classmethod
    def parse(cls, data: object) -> '_NodeConfig':
        required = {'name', 'area'}
        if (not isinstance(data, Mapping) or required - data.keys()
                or data.keys() - required - {'machine_tag'}):
            raise ValueError('Ogni nodo richiede name e area; machine_tag è opzionale')
        tag = nonempty_string(data['machine_tag'], 'node.machine_tag') if 'machine_tag' in data else None
        return cls(nonempty_string(data['name'], 'node.name'), nonempty_string(data['area'], 'node.area'), tag)


@dataclass(frozen=True)
class _RingConfig:
    """Capacità comuni e nodi appartenenti a un anello omogeneo."""

    id: str
    arch: str
    memory_mb: int
    cpus: float
    replicas: int
    nodes: tuple[_NodeConfig, ...]

    @classmethod
    def parse(cls, data: object) -> '_RingConfig':
        required = {'id', 'arch', 'memory_mb', 'cpus', 'nodes'}
        if (not isinstance(data, Mapping) or required - data.keys()
                or data.keys() - required - {'replicas'}):
            raise ValueError('Ogni ring richiede id, arch, memory_mb, cpus e nodes; replicas è opzionale')

        ring_id = nonempty_string(data['id'], 'ring.id')
        definitions = data['nodes']
        if not isinstance(definitions, list):
            raise ValueError(f'I nodi del ring {ring_id} devono essere una lista')

        nodes: dict[str, _NodeConfig] = {}
        for definition in definitions:
            node = _NodeConfig.parse(definition)
            if node.name in nodes:
                raise ValueError(f'Nodo già definito nel ring {ring_id}: {node.name}')
            nodes[node.name] = node

        return cls(
            ring_id,
            nonempty_string(data['arch'], 'ring.arch'),
            positive_integer(data['memory_mb'], 'ring.memory_mb'),
            positive_number(finite_time(data['cpus'], 'ring.cpus'), 'ring.cpus'),
            positive_integer(data.get('replicas', 128), 'ring.replicas'),
            tuple(nodes.values()),
        )


@dataclass(frozen=True)
class _InfrastructureConfig:
    """Topologia dichiarativa dalla quale ogni replica costruisce i suoi nodi."""

    rings: tuple[_RingConfig, ...]

    @classmethod
    def parse(cls, infrastructure: object) -> '_InfrastructureConfig':
        if infrastructure is None:
            return cls(())
        if not isinstance(infrastructure, Mapping) or infrastructure.keys() != {'rings'}:
            raise ValueError('infrastructure deve contenere rings')
        definitions = infrastructure['rings']
        if not isinstance(definitions, list):
            raise ValueError('infrastructure.rings deve essere una lista')

        # L'ordine dichiarato è significativo per la selezione del primo ring.
        rings: dict[str, _RingConfig] = {}
        for definition in definitions:
            ring = _RingConfig.parse(definition)
            if ring.id in rings:
                raise ValueError(f'Ring già definito: {ring.id}')
            rings[ring.id] = ring
        return cls(tuple(rings.values()))


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
    def parse(cls, data: object) -> '_FunctionConfig':
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

        # Il runtime risolve l'immagine e i limiti di concorrenza del modello.
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

        return cls(
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
            tag_pattern=metadata.get('tag_pattern', ''),
            tar_function_code=metadata.get('tar_function_code', ''),
        )

    def build(self) -> Function:
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
