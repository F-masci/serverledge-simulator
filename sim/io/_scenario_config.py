"""Sezioni interne della configurazione di uno scenario."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sim.model.function import Function, FunctionProfile, Runtime
from sim.utils.validation import finite_time, integer, nonempty_string, positive_integer, positive_number, \
    nonnegative_number


@dataclass(frozen=True)
class _SchedulerConfig:
    """Parametri della politica di ammissione locale, condivisi dai nodi."""

    queue_capacity: int

    @classmethod
    def parse(cls, scheduler: object) -> '_SchedulerConfig':
        """Parse della configurazione dello scheduler."""

        # Se non è definito, assume un dizionario vuoto.
        if scheduler is None:
            scheduler = {}

        # Controlla che sia un dizionario e che contenga solo la chiave 'queue'.
        if not isinstance(scheduler, Mapping) or scheduler.keys() - {'queue'}:
            raise ValueError('scheduler accetta solo queue')
        queue = scheduler.get('queue', {})

        # Controlla che 'queue' sia un dizionario e che contenga solo la chiave 'capacity'.
        if not isinstance(queue, Mapping) or queue.keys() - {'capacity'}:
            raise ValueError('scheduler.queue accetta solo capacity')
        capacity = integer(queue.get('capacity', 0), 'scheduler.queue.capacity')

        # Restituisce un'istanza di _SchedulerConfig con la capacità della coda validata.
        return cls(queue_capacity=capacity)


@dataclass(frozen=True)
class _ContainersConfig:
    """Politica per i container idle e tempi di avvio per immagine Docker."""

    idle_timeout: float
    cleanup_interval: float
    startup_times: Mapping[str, float]

    @classmethod
    def parse(cls, containers: object) -> '_ContainersConfig':
        """Parse della configurazione dei container."""

        # Se non è definito, assume un dizionario vuoto.
        if containers is None:
            containers = {}

        # Controlla che sia un dizionario e che contenga solo le chiavi
        # - 'idle_timeout'
        # - 'cleanup_interval'
        # - 'startup_times'
        if not isinstance(containers, Mapping) or containers.keys() - {'idle_timeout', 'cleanup_interval', 'startup_times'}:
            raise ValueError('containers accetta solo idle_timeout, cleanup_interval e startup_times')
        idle_timeout = finite_time(containers.get('idle_timeout', 600.0), 'containers.idle_timeout')
        idle_timeout = nonnegative_number(idle_timeout, 'containers.idle_timeout')
        cleanup_interval = finite_time(containers.get('cleanup_interval', 30.0), 'containers.cleanup_interval')
        cleanup_interval = positive_number(cleanup_interval, 'containers.cleanup_interval')

        # Parse dei tempi di avvio per immagine Docker.
        startup_times = containers.get('startup_times', {})
        if not isinstance(startup_times, Mapping):
            raise ValueError('containers.startup_times deve associare immagini a durate')
        image_times = {}
        for image, duration in startup_times.items():
            image = nonempty_string(image, 'containers.startup_times (immagine)')
            duration = finite_time(duration, f'containers.startup_times.{image}')
            duration = nonnegative_number(duration, f'containers.startup_times.{image}')
            image_times[image] = duration

        # Restituisce un'istanza di _ContainersConfig con i parametri validati.
        return cls(idle_timeout, cleanup_interval, MappingProxyType(image_times))


@dataclass(frozen=True)
class _NodeConfig:
    """Identità del nodo; architettura e capacità sono definite dal ring."""

    name: str
    area: str
    machine_tag: str | None

    @classmethod
    def parse(cls, data: object) -> '_NodeConfig':
        """Parse della configurazione di un nodo."""
        required = {'name', 'area'}
        optional = {'machine_tag'}
        if not isinstance(data, Mapping) or required - data.keys() or data.keys() - required - optional:
            raise ValueError('Ogni nodo richiede name e area; machine_tag è opzionale')
        name = nonempty_string(data['name'], 'node.name')
        area = nonempty_string(data['area'], 'node.area')
        tag = nonempty_string(data['machine_tag'], 'node.machine_tag') if 'machine_tag' in data else None

        # Restituisce un'istanza di _NodeConfig con i parametri validati.
        return cls(name, area, tag)


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
        """Parse della configurazione di un ring."""

        # Controlla le chiavi obbligatorie e il numero opzionale di repliche.
        required = {'id', 'arch', 'memory_mb', 'cpus', 'nodes'}
        optional = {'replicas'}
        if not isinstance(data, Mapping) or required - data.keys() or data.keys() - required - optional:
            raise ValueError('Ogni ring richiede id, arch, memory_mb, cpus e nodes; replicas è opzionale')

        # Legge i nodi dichiarati nel ring e verifica che i nomi siano distinti.
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

        # Valida le risorse condivise da tutti i nodi e le repliche virtuali.
        arch = nonempty_string(data['arch'], 'ring.arch')
        memory_mb = positive_integer(data['memory_mb'], 'ring.memory_mb')
        cpus = finite_time(data['cpus'], 'ring.cpus')
        cpus = positive_number(cpus, 'ring.cpus')
        replicas = positive_integer(data.get('replicas', 128), 'ring.replicas')

        # Restituisce un'istanza di _RingConfig con i parametri validati.
        return cls(ring_id, arch, memory_mb, cpus, replicas, tuple(nodes.values()))


@dataclass(frozen=True)
class _InfrastructureConfig:
    """Topologia dichiarativa dalla quale ogni replica costruisce i suoi nodi."""

    rings: tuple[_RingConfig, ...]

    @classmethod
    def parse(cls, infrastructure: object) -> '_InfrastructureConfig':
        """Parse della configurazione dell'infrastruttura."""

        # Se l'infrastruttura è omessa, lo scenario parte senza ring.
        if infrastructure is None:
            return cls(())

        # Controlla che la sezione contenga la lista dei ring.
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
        # Restituisce i ring validati nell'ordine dichiarato.
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
    def parse(
        cls, data: object, *, startup_times: Mapping[str, float] | None = None,
    ) -> '_FunctionConfig':
        """Parse della configurazione di una funzione."""

        # Controlla che la tabella contenga i campi richiesti e solo quelli ammessi.
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

        # Il runtime risolve l'immagine e i limiti di concorrenza del modello.
        runtime_name = nonempty_string(data['runtime'], 'runtime')
        custom_image = nonempty_string(data['custom_image'], 'custom_image') if 'custom_image' in data else None
        runtime = Runtime.get_runtime(runtime_name, custom_image=custom_image, startup_times=startup_times)

        # Se presenti, le architetture supportate devono formare una lista non vuota.
        supported_archs = None
        if 'supported_archs' in data:
            archs = data['supported_archs']
            if not isinstance(archs, list) or not archs:
                raise ValueError('supported_archs deve essere una lista non vuota')
            supported_archs = tuple(nonempty_string(arch, 'supported_archs') for arch in archs)

        # I metadati opzionali sono stringhe.
        metadata = {}
        for key in ('tag_pattern', 'tar_function_code'):
            if key in data:
                if not isinstance(data[key], str):
                    raise ValueError(f'{key} deve essere una stringa')
                metadata[key] = data[key]

        # Valida l'identità, le risorse e il limite di concorrenza.
        name = nonempty_string(data['name'], 'name')
        memory_mb = positive_integer(data['memory_mb'], 'memory_mb')
        cpu_demand = finite_time(data['cpu_demand'], 'cpu_demand')
        max_concurrency = positive_integer(data.get('max_concurrency', 1), 'max_concurrency')

        # Costruisce il profilo temporale con le distribuzioni di default.
        profile = FunctionProfile(
            arrival_rate=data['arrival_rate'],
            service_rate=data['service_rate'],
            first_arrival_time=data['first_arrival_time'],
            arrival_distribution=data.get('arrival_distribution', 'poisson'),
            service_distribution=data.get('service_distribution', 'gamma'),
            service_scv=data.get('service_scv'),
            code_load_time=data.get('code_load_time', 0),
        )

        # Restituisce un'istanza di _FunctionConfig pronta per la registrazione.
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
