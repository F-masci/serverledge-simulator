"""Identità e risorse locali di un nodo Serverledge.

I contatori busy/warm rappresentano prenotazioni, non consumo fisico misurato.
"""

from .request import Request
from sim.utils.validation import finite_time, integer

from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from collections import deque
from collections.abc import Callable, Iterator, Mapping

from .function import Function
from .container import DockerContainer, ContainerState
from .pool import ContainerPool


@dataclass(frozen=True)
class NodeID:
    key: str
    area: str
    arch: str
    machine_tag: str


class Resources:
    """Prenotazioni locali; RAM in unità da 2**20 byte, CPU frazionarie.

    Le mutazioni sono coordinate
    dal nodo insieme ai pool; le proprietà pubbliche sono di sola lettura.
    """

    def __init__(self, total_memory: int, total_cpus: float) -> None:
        import math
        if total_memory < 1 or not math.isfinite(total_cpus) or total_cpus <= 0:
            raise ValueError('Capacità del nodo non valide')
        self._total_memory = total_memory
        self._total_cpus = total_cpus
        self._busy_pool_used_mem = 0
        self._warm_pool_used_mem = 0
        self._used_cpus = 0.0

    @property
    def total_memory(self) -> int:
        return self._total_memory

    @property
    def total_cpus(self) -> float:
        return self._total_cpus

    @property
    def busy_pool_used_mem(self) -> int:
        return self._busy_pool_used_mem

    @property
    def warm_pool_used_mem(self) -> int:
        return self._warm_pool_used_mem

    @property
    def used_cpus(self) -> float:
        return self._used_cpus

    @property
    def free_memory(self) -> int:
        return self.total_memory - self.busy_pool_used_mem - self.warm_pool_used_mem

    @property
    def available_memory(self) -> int:
        return self.total_memory - self.busy_pool_used_mem

    @property
    def available_cpus(self) -> float:
        return self.total_cpus - self.used_cpus

    def _reserve(self, function: Function, *, idle: bool) -> None:
        if idle:
            self._warm_pool_used_mem += function.memory_mb
        else:
            self._busy_pool_used_mem += function.memory_mb
            self._used_cpus += function.cpu_demand

    def _release(self, function: Function, *, idle: bool) -> None:
        if idle:
            self._warm_pool_used_mem -= function.memory_mb
        else:
            self._busy_pool_used_mem -= function.memory_mb
            self._used_cpus -= function.cpu_demand

    def _activate(self, function: Function) -> None:
        self._release(function, idle=True)
        self._reserve(function, idle=False)

    def _deactivate(self, function: Function) -> None:
        self._release(function, idle=False)
        self._reserve(function, idle=True)


class Node:
    """Identità e contabilità delle risorse di un singolo nodo simulato.

    Serverledge usa LocalNode e LocalResources globali perché ogni processo
    rappresenta un nodo. Nel simulatore ogni istanza possiede il proprio stato.
    """

    def __init__(
        self,
        name: str,
        memory: int,
        cpus: float,
        area: str,
        arch: str,
        machine_tag: str | None = None,
        *,
        idle_timeout: float = 600,
        queue_capacity: int = 0,
    ) -> None:
        self._idle_timeout = finite_time(idle_timeout, 'node.idle_timeout')
        if self._idle_timeout < 0:
            raise ValueError('node.idle_timeout deve essere non negativo')
        self._id = NodeID(
            key=name,
            area=area,
            arch=arch,
            machine_tag=arch if machine_tag is None else machine_tag,
        )
        self._resources = Resources(total_memory=memory, total_cpus=cpus)
        self._pools: dict[str, ContainerPool] = {}
        self._pending: dict[DockerContainer, bool] = {}
        self._queue_capacity = max(0, integer(queue_capacity, 'queue_capacity'))
        self._queue: deque[Request] = deque()
        self._container_ids = count(1)
        self.on_container_state_change: Callable[[Node, DockerContainer, ContainerState | None], None] | None = None
        self.on_dequeue: Callable[[Request], None] | None = None
        self.on_container_idle: Callable[[Node, DockerContainer], None] | None = None

    @property
    def id(self) -> NodeID:
        return self._id

    @property
    def resources(self) -> Resources:
        return self._resources

    @property
    def pools(self) -> Mapping[str, ContainerPool]:
        return MappingProxyType(self._pools)

    def iter_containers(self) -> Iterator[DockerContainer]:
        """Enumera i container in creazione e quelli già nei pool."""
        yield from self._pending
        for pool in self._pools.values():
            yield from pool

    def _pool(self, function: Function) -> ContainerPool:
        if function.name not in self._pools:
            self._pools[function.name] = ContainerPool(function)
        pool = self._pools[function.name]
        if pool.function != function:
            raise ValueError('Aggiornamento della funzione non supportato dal modello')
        return pool

    def acquire(self, function: Function, *, only_warm: bool = False) -> DockerContainer | None:
        """Riusa un container o riserva una creazione; None indica indisponibilità.

        Una creazione deve essere conclusa con creation_succeeded/creation_failed.
        """
        pool = self._pool(function)
        container = pool._reusable_busy() if function.max_concurrency > 1 else None
        if container is not None:
            container._acquire()
            return container
        if pool.idle:
            if self.resources.available_cpus < function.cpu_demand:
                return None
            container = pool._pop_idle()
            container._acquire()
            self.resources._activate(function)
            pool._add(container)
            self._notify_container_state(container, ContainerState.IDLE)
            return container
        return None if only_warm else self._reserve_creation(function, idle=False)

    def prewarm(self, function: Function) -> DockerContainer | None:
        """Riserva un container destinato al pool idle.

        """
        self._pool(function)
        return self._reserve_creation(function, idle=True)

    def _reserve_creation(self, function: Function, *, idle: bool) -> DockerContainer | None:
        if (self.resources.available_cpus < function.cpu_demand
                or self.resources.available_memory < function.memory_mb):
            return None
        deficit = function.memory_mb - self.resources.free_memory
        if deficit > 0:
            # Ordine deterministico del simulatore; la mappa Go non garantisce ordine.
            candidates = []
            reclaimed = 0
            for pool in self._pools.values():
                for container in pool.idle:
                    candidates.append(container)
                    reclaimed += container.function.memory_mb
                    if reclaimed >= deficit:
                        break
                if reclaimed >= deficit:
                    break
            if reclaimed < deficit:
                return None
            for container in candidates:
                self._destroy_idle(container)
        self.resources._reserve(function, idle=idle)
        container = DockerContainer(f'{self.id.key}/c{next(self._container_ids)}', function)
        self._pending[container] = idle
        self._notify_container_state(container, None)
        return container

    def creation_succeeded(self, container: DockerContainer, now: float, timeout: float | None = None) -> None:
        idle = self._pending.pop(container)
        container._created(idle=idle, now=now, timeout=self._idle_timeout if timeout is None else timeout)
        self._pool(container.function)._add(container)
        self._notify_container_state(container, ContainerState.CREATING)
        if idle:
            self._notify_container_idle(container)

    def creation_failed(self, container: DockerContainer) -> None:
        idle = self._pending.pop(container)
        self.resources._release(container.function, idle=idle)
        container._destroy()
        self._notify_container_state(container, ContainerState.CREATING)

    def complete(self, container: DockerContainer, now: float, timeout: float | None = None) -> None:
        pool = self._pools.get(container.function.name)
        if pool is None or container not in pool.busy:
            raise ValueError('Container non attivo su questo nodo')
        container._complete(now, self._idle_timeout if timeout is None else timeout)
        if container.state is ContainerState.IDLE:
            pool._remove_busy(container)
            pool._add(container)
            self.resources._deactivate(container.function)
            self._notify_container_state(container, ContainerState.BUSY)
            self._notify_container_idle(container)
        self.dispatch_waiting()

    def _notify_container_state(self, container: DockerContainer, previous: ContainerState | None) -> None:
        if self.on_container_state_change is not None:
            self.on_container_state_change(self, container, previous)

    def _notify_container_idle(self, container: DockerContainer) -> None:
        if self.on_container_idle is not None:
            self.on_container_idle(self, container)

    def _destroy_idle(self, container: DockerContainer) -> None:
        pool = self._pools[container.function.name]
        container._destroy()
        pool._remove_idle(container)
        self.resources._release(container.function, idle=True)
        self._notify_container_state(container, ContainerState.IDLE)

    def expire(self, now: float, *, container: DockerContainer | None = None) -> None:
        """Un passaggio del janitor, da invocare ai tempi stabiliti dal motore.

        La distruzione Docker è assunta riuscita in questo modello iniziale.
        """
        if container is not None:
            pool = self._pools.get(container.function.name)
            if pool is not None and container.is_expired(now) and container in pool.idle:
                self._destroy_idle(container)
            return

        for pool in self._pools.values():
            for container in pool._remove_expired(now):
                container._destroy()
                self.resources._release(container.function, idle=True)
                self._notify_container_state(container, ContainerState.IDLE)

    def submit(self, request: Request, *, reject_on_failure: bool = True) -> bool:
        """Tenta l'ammissione locale dopo la selezione di un ring compatibile.

        Riusa un busy sotto il limite di concorrenza, poi un idle, oppure
        prenota una nuova creazione. Il ring può rinviare il rifiuto definitivo
        per provare altri nodi. Una creazione resta CREATING fino
        alla notifica esplicita del suo esito, senza esecuzione automatica.
        """
        request._ensure_new()
        container = self.acquire(request.function)
        if container is None:
            return self.enqueue_or_reject(request) if reject_on_failure else False
        request._accept(self, container, warm=container.state is not ContainerState.CREATING)
        return True

    def enqueue_or_reject(self, request: Request) -> bool:
        """Mette in attesa sul nodo destinatario se resta posto nella FIFO."""
        if len(self._queue) >= self._queue_capacity:
            request._reject('out_of_resources', node=self)
            return False
        request._enqueue(self)
        self._queue.append(request)
        return True

    def dispatch_waiting(self) -> None:
        """Dopo un completamento ammette le richieste finché la testa può partire."""
        while self._queue:
            request = self._queue[0]
            container = self.acquire(request.function)
            if container is None:
                break
            self._dequeue(container)

    def _dequeue(self, container: DockerContainer) -> None:
        request = self._queue.popleft()
        request._accept(self, container, warm=container.state is not ContainerState.CREATING)
        if self.on_dequeue is not None:
            self.on_dequeue(request)

    def __repr__(self) -> str:
        return self.id.key
