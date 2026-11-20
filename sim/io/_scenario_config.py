"""Sezioni interne della configurazione di uno scenario."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from sim.model.function import Function, FunctionProfile, Runtime
from sim.model.offloading import SchedulingPolicy
from sim.utils.validation import finite_time, integer, nonempty_string, positive_integer, positive_number, \
    nonnegative_number


@dataclass(frozen=True)
class _SchedulerConfig:
    """Parametri della politica di ammissione locale, condivisi dai nodi."""

    queue_capacity: int
    policy: SchedulingPolicy
    fallback_local: bool
    offloading_cache_validity: int
    nearby_interval: int
    monitoring_interval: int

    @classmethod
    def parse(cls, scheduler: object) -> '_SchedulerConfig':
        """Parse della configurazione dello scheduler."""

        # Se non è definito, assume un dizionario vuoto.
        if scheduler is None:
            scheduler = {}

        allowed = {'queue', 'policy', 'fallback_local', 'offloading_cache_validity',
                   'nearby_interval', 'monitoring_interval'}
        if not isinstance(scheduler, Mapping) or scheduler.keys() - allowed:
            raise ValueError(f'Campi scheduler ammessi: {", ".join(sorted(allowed))}')
        queue = scheduler.get('queue', {})

        # Controlla che 'queue' sia un dizionario e che contenga solo la chiave 'capacity'.
        if not isinstance(queue, Mapping) or queue.keys() - {'capacity'}:
            raise ValueError('scheduler.queue accetta solo capacity')
        capacity = integer(queue.get('capacity', 0), 'scheduler.queue.capacity')

        return cls(
            queue_capacity=capacity,
            policy=SchedulingPolicy(nonempty_string(scheduler.get('policy', 'default'), 'scheduler.policy')),
            fallback_local=_boolean(scheduler.get('fallback_local', False), 'scheduler.fallback_local'),
            offloading_cache_validity=integer(
                scheduler.get('offloading_cache_validity', 60), 'scheduler.offloading_cache_validity',
            ),
            nearby_interval=positive_integer(scheduler.get('nearby_interval', 20), 'scheduler.nearby_interval'),
            monitoring_interval=positive_integer(
                scheduler.get('monitoring_interval', 30), 'scheduler.monitoring_interval',
            ),
        )


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f'{name} deve essere un booleano')
    return value


def _names(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f'{name} deve essere una lista di nomi')
    names = tuple(nonempty_string(item, name) for item in value)
    if len(names) != len(set(names)):
        raise ValueError(f'{name} contiene nomi duplicati')
    return names


@dataclass(frozen=True)
class _InvocationConfig:
    """Ingresso e autorizzazione all'offloading per gli arrivi generati."""

    entry_node: str | None = None
    entry_area: str | None = None
    can_offload: bool = False
    entry_region: str | None = None

    @classmethod
    def parse(cls, invocation: object) -> '_InvocationConfig':
        if invocation is None:
            return cls()
        if not isinstance(invocation, Mapping) or invocation.keys() - {
            'entry_node', 'entry_area', 'entry_region', 'can_offload',
        }:
            raise ValueError('invocation accetta solo entry_node, entry_area, entry_region e can_offload')
        node = nonempty_string(invocation['entry_node'], 'invocation.entry_node') if 'entry_node' in invocation else None
        area = nonempty_string(invocation['entry_area'], 'invocation.entry_area') if 'entry_area' in invocation else None
        region = nonempty_string(invocation['entry_region'], 'invocation.entry_region') if 'entry_region' in invocation else None
        return cls(node, area, _boolean(invocation.get('can_offload', False), 'invocation.can_offload'), region)


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
    policy: SchedulingPolicy | None
    remote_area: str | None
    fallback_local: bool | None
    nearest_neighbors: tuple[str, ...] | None
    remote_region: str | None

    @classmethod
    def parse(cls, data: object) -> '_NodeConfig':
        """Parse della configurazione di un nodo."""
        required = {'name', 'area'}
        optional = {'machine_tag', 'policy', 'remote_area', 'remote_region', 'fallback_local', 'nearest_neighbors'}
        if not isinstance(data, Mapping) or required - data.keys() or data.keys() - required - optional:
            raise ValueError(f'Ogni nodo richiede name e area. Campi opzionali: {", ".join(sorted(optional))}')
        name = nonempty_string(data['name'], 'node.name')
        area = nonempty_string(data['area'], 'node.area')
        tag = nonempty_string(data['machine_tag'], 'node.machine_tag') if 'machine_tag' in data else None

        policy = SchedulingPolicy(nonempty_string(data['policy'], 'node.policy')) if 'policy' in data else None
        remote = nonempty_string(data['remote_area'], 'node.remote_area') if 'remote_area' in data else None
        region = nonempty_string(data['remote_region'], 'node.remote_region') if 'remote_region' in data else None
        if remote is not None and region is not None:
            raise ValueError('Usare un solo campo tra node.remote_area e node.remote_region')
        fallback = _boolean(data['fallback_local'], 'node.fallback_local') if 'fallback_local' in data else None
        neighbors = _names(data['nearest_neighbors'], 'node.nearest_neighbors') if 'nearest_neighbors' in data else None
        if neighbors is not None and (len(neighbors) > 2 or name in neighbors):
            raise ValueError('node.nearest_neighbors ammette fino a due altri nodi della stessa area')
        return cls(name, area, tag, policy, remote, fallback, neighbors, region)


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
    load_balancer_areas: tuple[str, ...] = ()

    @classmethod
    def parse(cls, infrastructure: object) -> '_InfrastructureConfig':
        """Parse della configurazione dell'infrastruttura."""

        # Se l'infrastruttura è omessa, lo scenario parte senza ring.
        if infrastructure is None:
            return cls(())

        # Controlla che la sezione contenga la lista dei ring.
        if (not isinstance(infrastructure, Mapping) or 'rings' not in infrastructure
                or infrastructure.keys() - {'rings', 'load_balancer_areas'}):
            raise ValueError('infrastructure richiede rings e accetta load_balancer_areas')
        definitions = infrastructure['rings']
        if not isinstance(definitions, list):
            raise ValueError('infrastructure.rings deve essere una lista')

        # L'ordine dichiarato è significativo per il fallback tra ring.
        rings: dict[str, _RingConfig] = {}
        for definition in definitions:
            ring = _RingConfig.parse(definition)
            if ring.id in rings:
                raise ValueError(f'Ring già definito: {ring.id}')
            rings[ring.id] = ring
        # Restituisce i ring validati nell'ordine dichiarato.
        areas = _names(infrastructure.get('load_balancer_areas', []), 'infrastructure.load_balancer_areas')
        return cls(tuple(rings.values()), areas)


@dataclass(frozen=True)
class _RegionConfig:
    infrastructure: _InfrastructureConfig | None
    service_time_s: float = 0.0
    offload_latency_s: float = 0.0
    offload_latency_by_source_s: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def parse(cls, name: str, data: object) -> '_RegionConfig':
        prefix = f'regions.{name}'
        if not isinstance(data, Mapping) or data.keys() - {
            'mode', 'infrastructure', 'service_time_s', 'offload_latency_s', 'offload_latency_by_source_s',
        }:
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
        latencies = MappingProxyType(latency_by_source)
        mode = data.get('mode', 'simulated')
        if mode == 'simulated':
            if 'service_time_s' in data:
                raise ValueError(f'{prefix}.service_time_s richiede mode = "always_succeeds"')
            return cls(_InfrastructureConfig.parse(data.get('infrastructure')), offload_latency_s=latency,
                       offload_latency_by_source_s=latencies)
        if mode == 'always_succeeds':
            if 'infrastructure' in data:
                raise ValueError(f'{prefix}.infrastructure richiede mode = "simulated"')
            duration = nonnegative_number(data.get('service_time_s', 0), f'{prefix}.service_time_s')
            return cls(None, duration, latency, latencies)
        raise ValueError(f'{prefix}.mode deve essere simulated o always_succeeds')


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
