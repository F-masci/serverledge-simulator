"""Identità e risorse locali di un nodo Serverledge.

I contatori busy/warm rappresentano prenotazioni, non consumo fisico misurato.
"""
from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from itertools import count
from types import MappingProxyType
from typing import TYPE_CHECKING

from sim.utils.validation import integer, nonnegative_number
from .container import DockerContainer, ContainerState
from .function import Function
from .offloading import NodeScheduling, OffloadingScheduler
from .pool import ContainerPool
from .request import Request

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

    Le mutazioni sono coordinate dal nodo insieme ai pool;
    le proprietà pubbliche sono di sola lettura.
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
        """Prenota risorse per una funzione, nel pool busy o warm."""
        if idle:
            self._warm_pool_used_mem += function.memory_mb
        else:
            self._busy_pool_used_mem += function.memory_mb
            self._used_cpus += function.cpu_demand

    def release(self, function: Function, *, idle: bool) -> None:
        """Rilascia le risorse occupate da una funzione."""
        if idle:
            self._warm_pool_used_mem -= function.memory_mb
        else:
            self._busy_pool_used_mem -= function.memory_mb
            self._used_cpus -= function.cpu_demand

    def activate(self, function: Function) -> None:
        """Attiva una funzione, spostandola dal pool warm a quello busy."""
        self.release(function, idle=True)
        self.reserve(function, idle=False)

    def deactivate(self, function: Function) -> None:
        """Disattiva una funzione, spostandola dal pool busy a quello warm."""
        self.release(function, idle=False)
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
        # Identificativo univoco del nodo
        self._id = NodeID(
            key=name,
            arch=arch,
            machine_tag=arch if machine_tag is None else machine_tag,
        )
        # Infrastruttura
        self._ring = ring

        # Risorse locali e pool di container
        self._resources = Resources(total_memory=memory, total_cpus=cpus)
        self._pools: dict[str, ContainerPool] = {}
        self._pending: dict[DockerContainer, bool] = {}
        self._container_ids = count(1)
        self._idle_timeout = nonnegative_number(idle_timeout, 'node.idle_timeout')

        # Scheduler e coda FIFO
        self._scheduler = OffloadingScheduler.create(tuple, NodeScheduling())
        self._queue_capacity = max(0, integer(queue_capacity, 'queue_capacity'))
        self._queue: deque[Request] = deque()

        ### Callback ###
        self.on_container_created: Callable[[Node, DockerContainer], None] | None = None
        self.on_container_acquired: Callable[[Node, DockerContainer], None] | None = None
        self.on_container_destroyed: Callable[[Node, DockerContainer], None] | None = None
        self.on_container_state_changed: Callable[[Node, DockerContainer, ContainerState | None, ContainerState], None] | None = None

        self.on_container_idle: Callable[[Node, DockerContainer], None] | None = None
        self.on_container_busy: Callable[[Node, DockerContainer], None] | None = None

        self.on_request_submitted: Callable[[Node, Request], None] | None = None
        self.on_request_dequeued: Callable[[Node, Request], None] | None = None
        self.on_request_forwarded: Callable[[Node, Request, Region, Node | None], Node | None] | None = None

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

    ### Scheduling ###

    @property
    def scheduler(self) -> OffloadingScheduler:
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = value

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

    ### Resources ###

    def can_allocate(self, function: Function) -> bool:
        """Verifica se il nodo può allocare risorse per la funzione."""
        return (
            self.resources.available_cpus >= function.cpu_demand and
            self.resources.available_memory >= function.memory_mb
        )

    #################################
    ### Internal state management ###
    #################################

    ## Request management ##

    def receive(self, request: Request) -> Node | None:
        """Applica la policy locale e inoltra la richiesta se necessario."""
        request.ensure_created()

        # Si delega la decisione di routing allo scheduler configurato sul nodo.
        #
        # Se lo scheduler restituisce un nodo o una regione, la richiesta viene inoltrata.
        # Se lo scheduler restituisce None, la richiesta può essere accettata o messa in coda.
        #
        # Lo scheduler esegue la submit locale se la policy lo consente,
        # altrimenti rifiuta la richiesta.
        target: Node | Region | None = self._scheduler.schedule(request, self)

        # Se lo scheduler ha scelto un target, si inoltra la richiesta:
        # - verso un altro nodo della stessa regione tramite l'infrastruttura;
        # - verso una regione remota tramite _forward_region.
        if isinstance(target, Node):
            return self.infrastructure.forward(request, target, source=self)
        if target is not None:
            return self._forward_region(request, target)

        # Se lo scheduler non ha scelto un target, si verifica lo stato della richiesta
        # a seguito della submit locale. Se la richiesta è stata accettata o messa in coda,
        if request.is_accepted or request.is_queued:
            return self

        return None

    def _forward_region(self, request: Request, destination: Region) -> Node | None:
        """Avvia l'offloading della richiesta verso una regione remota.

        Verifica l'eventuale nodo destinatario in assenza di load balancer,
        aggiorna lo stato di trasferimento della richiesta e invoca la callback
        on_request_forwarded per la consegna o la pianificazione dell'evento.
        """
        source = request.region
        assert source is not None

        # Se la regione remota è simulata ma priva di load balancer, verifica subito
        # la presenza e la compatibilità del primo nodo registrato.
        target: Node | None = None
        if destination.infrastructure is not None and not destination.has_load_balancer:
            target = destination.first_registered_node()
            if target is None or not request.function.supports_arch(target.id.arch):
                request.reject('out_of_resources', node=self)
                return None

        # Marca l'offloading e avvia il trasferimento con latenza di sola andata.
        request.mark_offloaded(self)
        delay = destination.offload_latency_from(source.name) / 2
        request.start_transfer(destination, delay)

        # Delega alla callback registrata la notifica e la consegna/schedulazione.
        if self.on_request_forwarded is not None:
            return self.on_request_forwarded(self, request, destination, target)

        return None

    def submit(
            self, request: Request, *, allow_queue: bool = True, reject_on_failure: bool = True,
    ) -> bool:
        """Tenta l'ammissione locale dopo la selezione di un ring compatibile.

        Riusa un busy sotto il limite di concorrenza, poi un idle, oppure
        prenota una nuova creazione. Durante il routing, un tentativo fallito
        lascia la richiesta disponibile per il candidato successivo.
        Una creazione resta CREATING fino alla notifica esplicita del suo esito,
        senza esecuzione automatica.
        """
        request.ensure_created()

        # Si notifica l'invio della richiesta al nodo, prima di tentare l'ammissione.
        if self.on_request_submitted is not None:
            self.on_request_submitted(self, request)

        # Si tenta di acquisire un container. La policy decide se usare la FIFO.
        container = self.acquire(request.function)
        if container is None:
            if allow_queue:
                return self.enqueue(request, reject_on_failure=reject_on_failure)
            if reject_on_failure:
                request.reject('out_of_resources', node=self)
            return False

        # Se l'ammissione ha successo, si notifica l'accettazione della richiesta sul container.
        request.accept_on_container(container=container)
        return True

    def enqueue(self, request: Request, *, reject_on_failure: bool = True) -> bool:
        """Mette in attesa sul nodo destinatario se resta posto nella FIFO."""

        # Se la coda è piena, si rifiuta la richiesta se richiesto.
        if len(self._queue) >= self._queue_capacity:
            if reject_on_failure:
                request.reject('out_of_resources', node=self)
            return False

        # Se c'è posto nella coda, si registra la richiesta in attesa e si notifica l'accodamento.
        request.enqueue(self)
        self._queue.append(request)

        return True

    def dispatch_waiting(self) -> None:
        """Tenta di ammettere le richieste in attesa nella coda FIFO."""
        while self._queue:
            # Si tenta di ammettere la prima richiesta in coda; se non ci sono risorse, si esce.
            request = self._queue[0]
            container = self.acquire(request.function)
            if container is None:
                break

            # Rimuove la richiesta dalla coda, notifica l'estrazione e poi la ammette sul container
            # affinché RequestDequeued preceda sempre RequestAccepted.
            request = self._queue.popleft()
            if self.on_request_dequeued is not None:
                self.on_request_dequeued(self, request)
            request.accept_on_container(container)

    ## Container management ##

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

            # Si riattiva il container idle e si aggiornano le prenotazioni
            # prima della transizione affinché le callback osservino le risorse aggiornate.
            container = pool.pop_idle()
            self.resources.activate(function)
            container.acquire()
            pool.add(container)

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
        self._bind_container_callbacks(container)
        self._pending[container] = idle
        if self.on_container_state_changed is not None:
            self.on_container_state_changed(self, container, None, container.state)
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
        """Segnala che la creazione del container è riuscita e lo aggiunge al pool."""
        if container not in self._pending:
            raise ValueError('Container non in pending')

        # Se la creazione è riuscita, si rimuove il container dalla lista
        # pending e lo si aggiunge al pool.
        idle = self._pending.pop(container)
        container.created(idle=idle, now=now, timeout=self._idle_timeout)
        pool = self._get_pool(container.function)
        pool.add(container)

    def complete(self, container: DockerContainer, now: float) -> None:
        """Segnala il completamento di una richiesta e aggiorna lo stato del container."""

        # Se il container non è attivo o non appartiene al pool, si solleva un'eccezione.
        pool = self._get_pool(container.function)
        if pool is None or container not in pool.busy:
            raise ValueError('Container non attivo su questo nodo')

        # Se questa è l'ultima richiesta attiva, si aggiornano pool e prenotazioni
        # prima della transizione a IDLE affinché le callback osservino le risorse aggiornate.
        if container.requests_count == 1:
            pool.remove_busy(container)
            self.resources.deactivate(container.function)

        # Segna il completamento della richiesta e aggiorna lo stato del container.
        container.complete(now, self._idle_timeout)

        # Se il container è tornato idle, lo si registra nel pool idle.
        #
        # TODO: spostare la logica di inversione della lista nell callback on_idle del container
        if container.is_idle:
            pool.add(container)

        # Dopo il completamento, si tenta di ammettere le richieste in attesa nella coda FIFO.
        self.dispatch_waiting()

    def expire(self, now: float, *, container: DockerContainer) -> None:
        """Rimuove il container indicato se è ancora nel pool idle ed è scaduto."""
        pool = self._get_pool(container.function)

        # Se il container non è scaduto o non è nel pool idle, non si fa nulla.
        if not container.is_expired(now): return
        if container not in pool.idle: return

        # Se il container è scaduto e nel pool idle, lo si distrugge e si aggiornano le prenotazioni.
        self._destroy_idle(container)

    def _destroy_idle(self, container: DockerContainer) -> None:
        """Rimuove un container idle dal pool e aggiorna le prenotazioni."""
        pool = self._get_pool(container.function)
        pool.remove_idle(container)
        self.resources.release(container.function, idle=True)
        container.destroy()

    ## Callback handling ##

    def _bind_container_callbacks(self, container: DockerContainer) -> None:
        """Associa le callback del nodo a un container."""
        on_created = self.on_container_created
        on_acquire = self.on_container_acquired
        on_destroyed = self.on_container_destroyed
        on_container_state_changed = self.on_container_state_changed

        on_idle = self.on_container_idle
        on_busy = self.on_container_busy

        if on_created is not None:
            container.on_created = partial(on_created, self)
        if on_acquire is not None:
            container.on_acquire = partial(on_acquire, self)
        if on_destroyed is not None:
            container.on_destroyed = partial(on_destroyed, self)
        if on_container_state_changed is not None:
            container.on_state_changed = partial(on_container_state_changed, self)

        if on_idle is not None:
            container.on_idle = partial(on_idle, self)
        if on_busy is not None:
            container.on_busy = partial(on_busy, self)

    #####################
    ### Serialization ###
    #####################

    def to_dict(self) -> dict:
        """Restituisce lo stato locale del nodo in un formato serializzabile."""
        resources = self.resources
        ring = self._ring
        return {
            'id': self.id.key,
            'arch': self.id.arch,
            'machine_tag': self.id.machine_tag,
            'ring_id': ring.id if ring is not None else None,
            'idle_timeout_s': self._idle_timeout,
            'resources': {
                'total_memory_mb': resources.total_memory,
                'busy_pool_used_memory_mb': resources.busy_pool_used_mem,
                'warm_pool_used_memory_mb': resources.warm_pool_used_mem,
                'free_memory_mb': resources.free_memory,
                'available_memory_mb': resources.available_memory,
                'total_cpus': resources.total_cpus,
                'used_cpus': resources.used_cpus,
                'available_cpus': resources.available_cpus,
            },
            'pools': {name: pool.to_dict() for name, pool in self._pools.items()},
            'pending': [
                {**container.to_dict(), 'prewarm': idle}
                for container, idle in self._pending.items()
            ],
            'queue_capacity': self._queue_capacity,
            'queue': [request.id for request in self._queue],
        }

    def to_json(self) -> str:
        """Serializza il nodo per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return self.id.key
