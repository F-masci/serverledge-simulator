"""Identità e risorse locali di un nodo Serverledge.

I contatori busy/warm rappresentano prenotazioni, non consumo fisico misurato.
"""
from __future__ import annotations
from .request import Request
from sim.utils.validation import finite_time, integer

from dataclasses import dataclass
from itertools import count
from types import MappingProxyType
from typing import TYPE_CHECKING
from collections import deque
from collections.abc import Callable, Iterator, Mapping

from .function import Function
from .container import DockerContainer, ContainerState
from .pool import ContainerPool

if TYPE_CHECKING:
    from .region import Region
    from .infrastructure import Infrastructure
    from .ring import ArchitectureRing


@dataclass(frozen=True)
class NodeID:
    key: str
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

    def reserve(self, function: Function, *, idle: bool) -> None:
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

    def activate(self, function: Function) -> None:
        self._release(function, idle=True)
        self.reserve(function, idle=False)

    def deactivate(self, function: Function) -> None:
        self._release(function, idle=False)
        self.reserve(function, idle=True)


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
        arch: str,
        machine_tag: str | None = None,
        *,
        idle_timeout: float = 600,
        queue_capacity: int = 0,
        ring: 'ArchitectureRing | None' = None,
    ) -> None:
        self._ring = ring
        self._idle_timeout = finite_time(idle_timeout, 'node.idle_timeout')
        if self._idle_timeout < 0:
            raise ValueError('node.idle_timeout deve essere non negativo')
        self._id = NodeID(
            key=name,
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

    ### Identification ###

    @property
    def id(self) -> NodeID:
        return self._id

    @property
    def key(self) -> str:
        return self.id.key

    ### Infrastructure ###

    @property
    def ring(self) -> ArchitectureRing:
        assert self._ring is not None, 'Nodo non associato a un anello'
        return self._ring

    @property
    def infrastructure(self) -> 'Infrastructure':
        return self.ring.infrastructure

    @property
    def region(self) -> Region:
        return self.infrastructure.region

    ### Node State ###

    @property
    def resources(self) -> Resources:
        return self._resources

    @property
    def pools(self) -> Mapping[str, ContainerPool]:
        return MappingProxyType(self._pools)

    @property
    def containers(self) -> Mapping[str, tuple[DockerContainer, ...]]:
        """Restituisce i container suddivisi per stato."""
        return MappingProxyType({
            "pending": tuple(self._pending),
            "created": tuple(
                container
                for pool in self._pools.values()
                for container in pool
            ),
        })

    def _get_pool(self, function: Function) -> ContainerPool:
        if function.name not in self._pools:
            self._pools[function.name] = ContainerPool(function)
        pool = self._pools[function.name]
        if pool.function != function:
            raise ValueError('Aggiornamento della funzione non supportato dal modello')
        return pool

    #################################
    ### Internal state management ###
    #################################

    ## Container selection ##

    def acquire(self, function: Function) -> DockerContainer | None:
        """Riusa un container o riserva una creazione; None indica indisponibilità.

        Una creazione viene conclusa con creation_succeeded.
        """
        pool = self._get_pool(function)

        # Prima si cerca un container già in uso
        # che non abbia raggiunto la concorrenza massima.
        container = pool.get_reusable_container()

        # Se non c'è un container riutilizzabile, si tenta di crearne uno nuovo.
        if container is None:
            return self._reserve_creation(function, idle=False)

        # Se il container è busy, lo si riutilizza senza modificare le prenotazioni.
        if container.is_busy:
            container.acquire()
            return container

        # Se il container è idle, lo si riattiva e si aggiornano le prenotazioni.
        if container.is_idle:

            # Se non ci sono CPU disponibili, non si può riattivare il container.
            if self.resources.available_cpus < function.cpu_demand:
                return None

            # Si riattiva il container idle e si aggiornano le prenotazioni.
            container = pool.pop_idle()
            container.acquire()
            self.resources.activate(function)
            pool.add(container)
            self._notify_container_state(container, ContainerState.IDLE)

            return container

        # Se il container non è né busy né idle, non è riutilizzabile.
        return None

    def prewarm(self, function: Function) -> DockerContainer | None:
        """Riserva un container destinato al pool idle."""
        self._get_pool(function)
        return self._reserve_creation(function, idle=True)

    def _reserve_creation(self, function: Function, *, idle: bool) -> DockerContainer | None:
        """
        Riserva la creazione di un nuovo container, se le risorse lo consentono

        Se idle è True, il container sarà destinato al pool idle; altrimenti sarà destinato a un uso immediato.
        Restituisce il container riservato o None se le risorse non sono sufficienti.
        Viene effettuata una verifica delle risorse disponibili e, se necessario, vengono distrutti
        container idle per liberare memoria.
        """
        if self.resources.available_cpus < function.cpu_demand: return None
        if self.resources.available_memory < function.memory_mb: return None

        # Se non ci sono abbastanza risorse disponibili, si cerca di liberare memoria.
        deficit = function.memory_mb - self.resources.free_memory
        if deficit > 0 and self._free_memory(deficit) < deficit:
            # Se non è possibile liberare abbastanza memoria, la creazione non può essere riservata.
            return None

        # Se le risorse sono sufficienti, si riserva la creazione del container e si aggiorna lo stato.
        self.resources.reserve(function, idle=idle)
        container = DockerContainer(f'{self.id.key}/c{next(self._container_ids)}', function, node=self)
        self._pending[container] = idle
        self._notify_container_state(container, None)
        return container

    def _free_memory(self, to_free: int) -> int:
        """Libera la memoria richiesta o restituisce zero se gli idle non bastano."""
        if to_free <= 0:
            return 0

        candidates = []
        reclaimable = 0

        # Raccoglie i candidati nell'ordine dei pool e delle rispettive liste idle.
        for pool in self._pools.values():
            for container in pool.idle:
                candidates.append(container)
                reclaimable += container.function.memory_mb
                if reclaimable >= to_free:
                    break
            if reclaimable >= to_free:
                break

        # Conserva tutti i container warm se il loro rilascio non copre il deficit.
        if reclaimable < to_free:
            return 0

        # Rimuove i candidati e aggiorna le prenotazioni del nodo.
        for container in candidates:
            self._destroy_idle(container)

        return reclaimable

    def creation_succeeded(self, container: DockerContainer, now: float) -> None:
        idle = self._pending.pop(container)
        container.created(idle=idle, now=now, timeout=self._idle_timeout)
        self._get_pool(container.function).add(container)
        self._notify_container_state(container, ContainerState.CREATING)
        if idle:
            self._notify_container_idle(container)

    def complete(self, container: DockerContainer, now: float) -> None:
        pool = self._pools.get(container.function.name)
        if pool is None or container not in pool.busy:
            raise ValueError('Container non attivo su questo nodo')
        container.complete(now, self._idle_timeout)
        if container.state is ContainerState.IDLE:
            pool.remove_busy(container)
            pool.add(container)
            self.resources.deactivate(container.function)
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
        container.destroy()
        pool.remove_idle(container)
        self.resources._release(container.function, idle=True)
        self._notify_container_state(container, ContainerState.IDLE)

    def expire(self, now: float, *, container: DockerContainer) -> None:
        """Rimuove il container indicato se è ancora nel pool idle ed è scaduto."""
        pool = self._pools.get(container.function.name)
        if pool is not None and container.is_expired(now) and container in pool.idle:
            self._destroy_idle(container)

    def submit(self, request: Request, *, reject_on_failure: bool = True) -> bool:
        """Tenta l'ammissione locale dopo la selezione di un ring compatibile.

        Riusa un busy sotto il limite di concorrenza, poi un idle, oppure
        prenota una nuova creazione. Durante il routing, un tentativo fallito
        lascia la richiesta disponibile per il candidato successivo.
        Una creazione resta CREATING fino
        alla notifica esplicita del suo esito, senza esecuzione automatica.
        """
        request.ensure_created()
        container = self.acquire(request.function)
        if container is None:
            return self.enqueue_or_reject(request) if reject_on_failure else False
        request.accept_on_container(container=container)
        return True

    def enqueue_or_reject(self, request: Request) -> bool:
        """Mette in attesa sul nodo destinatario se resta posto nella FIFO."""
        if len(self._queue) >= self._queue_capacity:
            request.reject('out_of_resources', node=self)
            return False
        request.enqueue(self)
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
        request.accept_on_container(self, container)
        if self.on_dequeue is not None:
            self.on_dequeue(request)

    def __repr__(self) -> str:
        return self.id.key
