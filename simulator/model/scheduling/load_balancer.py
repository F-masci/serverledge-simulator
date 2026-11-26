"""Strategie di selezione dell'anello di destinazione per le invocazioni."""

import json
from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Never

from simulator.random.streams import RandomStreams, StreamKind
from simulator.utils.validation import positive_integer, positive_number

from ..execution.function import Function
from ..execution.request import Request, RequestStatus
from ..topology.node import Node
from ..topology.ring import ArchitectureRing


class LoadBalancerMode(StrEnum):
    """Strategie di bilanciamento del carico selezionabili nello scenario."""
    RANDOM = 'Random'                             # Selezione casuale uniforme tra gli anelli compatibili.
    ROUND_ROBIN = 'RoundRobin'                    # Selezione circolare tra gli anelli compatibili.
    MAB = 'MAB'                                   # Multi-Armed Bandit (non ancora disponibile nel simulatore).
    ARCHITECTURE_UNAWARE = 'ArchitectureUnaware'  # Singolo hash ring globale indipendente da architettura e machine_tag.


@dataclass
class NodeMetric:
    """Stato delle risorse di un nodo memorizzato nella cache locale del load balancer."""

    total_memory_mb: int
    free_memory_mb: int
    total_cpus: float
    free_cpus: float
    last_update_s: float = 0.0


class NodeMetricCache:
    """
    Cache locale delle risorse dei nodi gestita dal reverse proxy Serverledge.

    ---

    Nel sistema reale, il load balancer non conosce in tempo reale le risorse esatte dei worker,
    ma mantiene una cache locale ``NodeMetrics`` con il seguente ciclo di vita:

    1. **Inizializzazione**:
       quando un worker viene scoperto, il LB interroga ``/status`` e popola``FreeMemoryMB``
       e ``FreeCPU``.
    2. **Ammissione preventiva**:
       prima di inoltrare la richiesta al worker, il LB filtra i candidati controllando
       ``FreeMemoryMB >= fun.MemoryMB`` e ``FreeCPU >= fun.CPUDemand`` sulla propria cache.
       Se nessun nodo nei ring compatibili soddisfa il controllo, il LB rifiuta subito la
       richiesta (HTTP 429) senza consegnarla al worker, impedendo in quel caso sia
       l'accodamento locale sia l'eventuale offload gestito dallo scheduler del worker.
    3. **Decremento pessimistico all'invio**:
       appena selezionato il nodo candidato, il LB sottrae immediatamente ``fun.MemoryMB``
       e ``fun.CPUDemand`` dalla cache locale. Questo avviene **sempre**, anche quando la
       richiesta sul worker sarà servita da un container warm già allocato (che in realtà
       non consuma nuova RAM) oppure verrà inoltrata in offload (``cloudonly`` / ``edgecloud``).
    4. **Riconciliazione su risposta e polling periodico**:
       la cache viene riallineata ai valori effettivi del worker in due momenti:
       - al ritorno della risposta HTTP (tramite gli header ``Serverledge-Free-Mem`` e
         ``Serverledge-Free-CPU``);
       - nel ciclo di polling periodico ``/status``, ogni ``load_balancer_refresh_interval_s``
         secondi (default 30 s, come ``lb.refresh_interval`` in Serverledge).

    La modalità è attiva di default (``load_balancer_resource_cache = true``) per riflettere
    fedelmente il comportamento del sistema reale, ma può essere disattivata da configurazione
    (``load_balancer_resource_cache = false``) per confrontarlo con un controllo istantaneo
    basato sullo stato corrente dei nodi (``node.can_allocate``).
    """

    def __init__(self) -> None:
        self._metrics: dict[str, NodeMetric] = {}

    def register(self, node: Node, *, now: float = 0.0) -> None:
        """Registra un nodo nella cache leggendone la disponibilità iniziale."""
        resources = node.resources
        self._metrics[node.key] = NodeMetric(
            total_memory_mb=resources.total_memory,
            free_memory_mb=resources.available_memory,
            total_cpus=resources.total_cpus,
            free_cpus=resources.available_cpus,
            last_update_s=now,
        )

    def remove(self, node: Node) -> None:
        """Rimuove un nodo dalla cache."""
        self._metrics.pop(node.key, None)

    def get(self, node: Node) -> NodeMetric | None:
        """Restituisce la voce di cache del nodo, se presente."""
        return self._metrics.get(node.key)

    def has_enough_resources(self, node: Node, function: Function) -> bool:
        """
        Verifica CPU e memoria dalla cache locale.
        Se il nodo non è ancora presente in cache l'ammissione è ottimistica.
        """
        metric = self._metrics.get(node.key)

        # Se il nodo non è ancora registrato nella cache, consideriamo che abbia risorse sufficienti.
        if metric is None:
            return True

        # Verifica se il nodo ha abbastanza memoria e CPU libera per la funzione richiesta.
        return (
            metric.free_memory_mb >= function.memory_mb
            and metric.free_cpus >= function.cpu_demand
        )

    def reserve(self, node: Node, function: Function, *, now: float = 0.0) -> None:
        """
        Applica il decremento pessimistico prima di inoltrare la richiesta al worker.

        Sottrae sempre ``memory_mb`` e ``cpu_demand`` dalla voce in cache, replicando il
        comportamento del sistema reale.
        """
        metric = self._metrics.get(node.key)

        # Se il nodo non è ancora registrato nella cache, lo registriamo con le risorse correnti.
        if metric is None:
            self.register(node, now=now)
            metric = self._metrics[node.key]

        # Decrementa le risorse libere nella cache locale.
        metric.free_memory_mb -= function.memory_mb
        metric.free_cpus -= function.cpu_demand

        # Aggiorna il timestamp dell'ultima modifica solo se il nuovo timestamp è più recente.
        if now >= metric.last_update_s:
            metric.last_update_s = now

    def reconcile(self, node: Node, *, now: float = 0.0) -> None:
        """
        Riallinea la cache ai valori reali del nodo.

        Ignora aggiornamenti più vecchi dell'ultimo timestamp registrato.
        """
        metric = self._metrics.get(node.key)

        # Se il nodo non è ancora registrato nella cache o l'aggiornamento è più vecchio dell'ultimo, non facciamo nulla.
        if metric is not None and now < metric.last_update_s:
            return

        # Aggiorna la voce di cache con le risorse correnti del nodo.
        resources = node.resources
        self._metrics[node.key] = NodeMetric(
            total_memory_mb=resources.total_memory,
            free_memory_mb=resources.available_memory,
            total_cpus=resources.total_cpus,
            free_cpus=resources.available_cpus,
            last_update_s=now,
        )

    def refresh_all(self, nodes: Iterable[Node], *, now: float = 0.0) -> None:
        """Esegue il polling periodico /status su tutti i nodi attivi della regione."""
        active_keys: set[str] = set()
        for node in nodes:
            active_keys.add(node.key)
            self.reconcile(node, now=now)

        # Rimuove i nodi non più attivi dalla cache.
        for stale_key in list(self._metrics.keys() - active_keys):
            del self._metrics[stale_key]


class LoadBalancer(ABC):
    """
    Contratto comune delle strategie di bilanciamento tra anelli di una regione.

    Filtra gli anelli compatibili con la funzione richiesta (non vuoti, architettura
    supportata e compatibilità con il machine tag) e determina l'ordine
    con cui l'infrastruttura tenta la consegna dell'invocazione.

    Quando ``resource_cache=True`` (default, fedele al sistema reale),
    il bilanciatore gestisce una ``NodeMetricCache`` locale con decremento pessimistico
    all'invio e riconciliazione su risposta/polling periodico. Quando ``resource_cache=False``,
    verifica invece in tempo reale ``node.can_allocate(function)``.
    """

    mode: ClassVar[LoadBalancerMode]

    def __init__(
        self,
        *,
        resource_cache: bool = True,
        refresh_interval_s: float = 30.0,
    ) -> None:
        # La cache risorse del proxy reale è attiva per default.
        self._resource_cache = resource_cache
        self._refresh_interval_s = positive_number(refresh_interval_s, 'refresh_interval_s')
        self._metric_cache = NodeMetricCache()
        # Traccia il nodo di ingresso selezionato dal proxy per ogni richiesta in volo,
        # così da riconciliare quel nodo al ritorno della risposta.
        self._in_flight: dict[str, Node] = {}
        # Ordine di prima attivazione dei ring (un ring entra nell'ordine solo quando riceve il primo nodo).
        self._discovered: list[ArchitectureRing] = []

    @staticmethod
    def create(
        mode: LoadBalancerMode,
        *,
        seed: int,
        stream_id: str = 'load_balancer',
        replicas: int = 128,
        resource_cache: bool = True,
        refresh_interval_s: float = 30.0,
    ) -> 'LoadBalancer':
        """Costruisce il load balancer corrispondente alla modalità configurata nello scenario."""
        if mode is LoadBalancerMode.RANDOM:
            return RandomLoadBalancer(
                seed=seed, stream_id=stream_id,
                resource_cache=resource_cache, refresh_interval_s=refresh_interval_s,
            )

        if mode is LoadBalancerMode.ROUND_ROBIN:
            return RoundRobinLoadBalancer(
                resource_cache=resource_cache, refresh_interval_s=refresh_interval_s,
            )

        if mode is LoadBalancerMode.ARCHITECTURE_UNAWARE:
            return ArchitectureUnawareLoadBalancer(
                replicas=replicas,
                resource_cache=resource_cache, refresh_interval_s=refresh_interval_s,
            )

        # Le strategie non ancora modellate sollevano NotImplementedError in fase di costruzione.
        return _UNAVAILABLE_LOAD_BALANCERS[mode]()

    #################################
    ### Internal state management ###
    #################################

    ## Infrastructure state ##

    def register_node(self, ring: ArchitectureRing, node: Node | None = None, *, now: float = 0.0) -> None:
        """Registra il ring nell'ordine di scoperta e inizializza il nodo nella cache."""

        # Se il ring non è ancora stato scoperto, lo aggiungiamo all'elenco dei ring scoperti.
        if ring not in self._discovered:
            self._discovered.append(ring)

        # Se la cache risorse è attiva e il nodo è fornito, lo registriamo nella cache.
        if self._resource_cache and node is not None:
            self._metric_cache.register(node, now=now)

    def unregister_node(self, node: Node) -> None:
        """Rimuove il nodo dalla cache risorse del bilanciatore."""
        if self._resource_cache:
            self._metric_cache.remove(node)

    ## Ring selection ##

    def route(
        self, request: Request, rings: Iterable[ArchitectureRing], *, now: float = 0.0,
    ) -> Node | None:
        """Instrada l'invocazione sugli anelli compatibili e gestisce il fallback."""
        candidates = self.candidate_rings(request, rings)
        function = request.function

        # Seleziona il primo nodo ammissibile tra gli anelli candidati e inoltra la richiesta.
        for ring in candidates:
            is_eligible = lambda candidate: self._is_node_eligible(candidate, function)
            node = ring.select_eligible_node(function, is_eligible=is_eligible)

            # Se non ci sono nodi ammissibili nell'anello, passa al successivo.
            if node is None:
                continue

            # Inoltra la richiesta al nodo selezionato e aggiorna la cache risorse se attiva.
            destination = self._dispatch_to_node(node, request, now=now)
            if request.status is not RequestStatus.CREATED:
                # La risposta dello scheduler arriva dopo la selezione del proxy.
                return destination

        request.reject('out_of_resources' if candidates else 'no_destination')
        return None

    def _is_node_eligible(self, node: Node, function: Function) -> bool:
        """Valuta l'ammissibilità del nodo secondo la modalità configurata (cache reale vs istantanea)."""
        if self._resource_cache:
            return self._metric_cache.has_enough_resources(node, function)

        return node.can_allocate(function)

    def _dispatch_to_node(self, node: Node, request: Request, *, now: float) -> Node | None:
        """
        Inoltra la richiesta al nodo scelto applicando, se attiva, la semantica della cache reale.

        In modalità ``resource_cache=True``, replica la sequenza del sistema reale:
        decremento pessimistico → proxy verso il worker → riconciliazione immediata
        solo se la risposta torna subito (rifiuto/completamento sincrono), altrimenti differita
        al completamento/rifiuto asincrono tramite ``reconcile_request``.
        """
        if self._resource_cache:
            self._metric_cache.reserve(node, request.function, now=now)
            self._in_flight[request.id] = node

        destination = node.receive(request, now=now)

        # Se il worker ha già prodotto un esito sincrono (es. rifiuto immediato senza coda),
        # la risposta HTTP ritorna subito al proxy e aggiorna la cache.
        if self._resource_cache and (request.is_completed or request.is_rejected):
            self.reconcile_request(request, now=now)

        return destination

    def submit(
        self, request: Request, rings: Iterable[ArchitectureRing],
    ) -> ArchitectureRing | None:
        """Restituisce l'anello primario selezionato per l'invocazione, o None se assente."""
        return next(iter(self.candidate_rings(request, rings)), None)

    def candidate_rings(
        self, request: Request, rings: Iterable[ArchitectureRing],
    ) -> tuple[ArchitectureRing, ...]:
        """Restituisce l'anello primario seguito dagli altri anelli compatibili in ordine di scoperta."""
        function = request.function
        current = tuple(rings)
        current_set = set(current)

        # Fallback lazy per usi isolati del balancer fuori da Infrastructure:
        # registra eventuali ring non vuoti non ancora notificati tramite register_node.
        self._discovered = [ring for ring in self._discovered if ring in current_set]
        for ring in current:
            if len(ring) and ring not in self._discovered:
                self._discovered.append(ring)

        registered = tuple(self._discovered)

        # Filtra gli anelli non vuoti compatibili per architettura e machine_tag
        # (1 ring = 1 machine_tag, come compatibleTagsForFunction in generalLoadBalancer.go).
        compatible = tuple(
            ring for ring in registered
            if len(ring)
            and function.supports_arch(ring.arch)
            and function.supports_machine_tag(ring.machine_tag)
        )
        if not compatible:
            return ()

        # Seleziona l'anello primario secondo la strategia concreta e mantiene
        # i restanti anelli compatibili come candidati di fallback.
        index = self._select_index(request, compatible, registered)
        return compatible[index], *compatible[:index], *compatible[index + 1:]

    @abstractmethod
    def _select_index(
        self,
        request: Request,
        compatible: tuple[ArchitectureRing, ...],
        registered: tuple[ArchitectureRing, ...],
    ) -> int:
        """Sceglie l'indice dell'anello primario tra i candidati compatibili."""

    ## Resource cache management ##

    def reconcile_request(self, request: Request, *, now: float = 0.0) -> None:
        """Riconcilia la cache del nodo di ingresso al termine della richiesta."""
        if not self._resource_cache:
            return

        # Recupera il nodo di ingresso associato alla richiesta e aggiorna la cache risorse.
        entry_node = self._in_flight.pop(request.id, None)
        if entry_node is not None:
            self._metric_cache.reconcile(entry_node, now=now)

    def refresh_metrics(self, nodes: Iterable[Node], *, now: float = 0.0) -> None:
        """Esegue il polling periodico /status sui nodi attivi (come refreshTargets in lb.go)."""
        if self._resource_cache:
            self._metric_cache.refresh_all(nodes, now=now)

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile del load balancer."""
        return f'{type(self).__name__}(mode={self.mode.value})'

    def to_dict(self) -> dict:
        """Restituisce la configurazione del load balancer in formato serializzabile."""
        return {
            'type': type(self).__name__,
            'mode': self.mode.value,
            'resource_cache': self._resource_cache,
            'refresh_interval_s': self._refresh_interval_s,
        }

    def to_json(self) -> str:
        """Serializza il load balancer per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} mode={self.mode.value}>'


class RandomLoadBalancer(LoadBalancer):
    """Seleziona casualmente l'anello primario tra quelli compatibili."""

    mode = LoadBalancerMode.RANDOM

    def __init__(
        self,
        *,
        seed: int = 12345,
        stream_id: str = 'load_balancer',
        resource_cache: bool = True,
        refresh_interval_s: float = 30.0,
    ) -> None:
        super().__init__(resource_cache=resource_cache, refresh_interval_s=refresh_interval_s)
        # Stream PRNG deterministico dedicato alle scelte di routing della regione.
        self._rng = RandomStreams(seed).stream(StreamKind.ROUTING, stream_id)

    #################################
    ### Internal state management ###
    #################################

    ## Ring selection ##

    def _select_index(
        self,
        request: Request,
        compatible: tuple[ArchitectureRing, ...],
        registered: tuple[ArchitectureRing, ...],
    ) -> int:
        """Estrae un indice uniforme tra gli anelli compatibili disponibili."""
        return int(self._rng.integers(len(compatible)))


class RoundRobinLoadBalancer(LoadBalancer):
    """
    Seleziona ciclicamente l'anello primario seguendo l'ordine di registrazione.

    Replica il comportamento di selectArchitectureRRFrom in Serverledge:
    mantiene un indice globale sugli anelli registrati e li scorre in modo
    circolare fino al primo anello compatibile con la richiesta.
    """

    mode = LoadBalancerMode.ROUND_ROBIN

    def __init__(
        self,
        *,
        resource_cache: bool = True,
        refresh_interval_s: float = 30.0,
    ) -> None:
        super().__init__(resource_cache=resource_cache, refresh_interval_s=refresh_interval_s)
        # Indice circolare sull'elenco degli anelli scoperti nella regione.
        self._rr_index = 0

    #################################
    ### Internal state management ###
    #################################

    ## Ring selection ##

    def _select_index(
        self,
        request: Request,
        compatible: tuple[ArchitectureRing, ...],
        registered: tuple[ArchitectureRing, ...],
    ) -> int:
        """Scorre gli anelli registrati da _rr_index fino al primo anello compatibile."""
        compatible_index = {ring: i for i, ring in enumerate(compatible)}
        total = len(registered)

        # Scorre in modo circolare gli anelli registrati, saltando quelli incompatibili.
        for offset in range(total):
            position = (self._rr_index + offset) % total
            ring = registered[position]
            if ring in compatible_index:
                # Avanza l'indice oltre l'anello scelto per la chiamata successiva.
                self._rr_index = (position + 1) % total
                return compatible_index[ring]

        raise RuntimeError('Nessun anello compatibile trovato tra gli anelli registrati')


class ArchitectureUnawareLoadBalancer(LoadBalancer):
    """
    Singolo anello di consistent hashing globale su tutti i nodi della regione.

    Replica ArchitectureUnawareBalancer di Serverledge (internal/lb/architectureUnawareLb.go):
    ignora machine_tag e tag_pattern, inserendo tutti i nodi in un unico hash ring FNV-1a
    a 128 repliche virtuali e filtrando i candidati solo per architettura supportata e risorse.
    """

    mode = LoadBalancerMode.ARCHITECTURE_UNAWARE

    def __init__(
        self,
        *,
        replicas: int = 128,
        resource_cache: bool = True,
        refresh_interval_s: float = 30.0,
    ) -> None:
        super().__init__(resource_cache=resource_cache, refresh_interval_s=refresh_interval_s)
        self._replicas = positive_integer(replicas, 'replicas')
        self._snapshot: tuple[Node, ...] = ()
        self._positions: list[int] = []
        self._targets: dict[int, list[Node]] = {}

    #################################
    ### Internal state management ###
    #################################

    ## Global ring management ##

    def _sync(self, rings: Iterable[ArchitectureRing]) -> None:
        """Allinea l'anello globale con l'insieme corrente dei nodi registrati."""
        nodes = tuple(node for ring in rings for node in ring)
        if nodes == self._snapshot:
            return

        positions: list[int] = []
        targets: dict[int, list[Node]] = {}
        for node in nodes:
            for index in range(self._replicas):
                position = ArchitectureRing._hash(f'{node.id.key}#{index}')
                if position not in targets:
                    positions.append(position)
                    targets[position] = []
                targets[position].append(node)
        positions.sort()

        self._snapshot = nodes
        self._positions = positions
        self._targets = targets

    def _candidates(self, function: Function, rings: Iterable[ArchitectureRing]) -> Iterator[Node]:
        """Visita i nodi compatibili per architettura in ordine di consistent hashing globale."""
        self._sync(rings)
        target_count = len(self._snapshot)
        size = len(self._positions)
        if target_count == 0 or size == 0:
            return

        start = bisect_left(self._positions, ArchitectureRing._hash(function.name))
        seen: set[Node] = set()
        for offset in range(size):
            node = self._targets[self._positions[(start + offset) % size]][-1]
            if node in seen:
                continue
            seen.add(node)
            if function.supports_arch(node.id.arch):
                yield node
            if len(seen) == target_count:
                break

    def select_node(self, function: Function, rings: Iterable[ArchitectureRing]) -> Node | None:
        """Restituisce il primo nodo compatibile per architettura nell'anello globale."""
        return next(self._candidates(function, rings), None)

    def _select_index(
        self,
        request: Request,
        compatible: tuple[ArchitectureRing, ...],
        registered: tuple[ArchitectureRing, ...],
    ) -> int:
        """Non usato nel percorso unaware: restituisce il primo anello compatibile."""
        return 0

    def route(
        self, request: Request, rings: Iterable[ArchitectureRing], *, now: float = 0.0,
    ) -> Node | None:
        """Instrada la richiesta sul singolo hash ring globale ignorando i machine_tag."""
        request.ensure_created()
        function = request.function
        candidates = tuple(self._candidates(function, rings))
        if not candidates:
            request.reject('no_destination')
            return None

        node = next((c for c in candidates if self._is_node_eligible(c, function)), None)
        if node is None:
            request.reject('out_of_resources')
            return None

        return self._dispatch_to_node(node, request, now=now)


class _UnavailableLoadBalancer(LoadBalancer):
    """Segnaposto per strategie Serverledge non ancora implementate nel simulatore."""

    mode: LoadBalancerMode

    def __init__(self) -> None:
        self._raise_unavailable()

    def _select_index(
        self,
        request: Request,
        compatible: tuple[ArchitectureRing, ...],
        registered: tuple[ArchitectureRing, ...],
    ) -> int:
        self._raise_unavailable()

    ######################
    ### Error handling ###
    ######################

    def _raise_unavailable(self) -> Never:
        """Segnala che la strategia selezionata non è supportata dal simulatore."""
        raise NotImplementedError(f'Load balancer {self.mode} non disponibile nel simulatore')


class MABLoadBalancer(_UnavailableLoadBalancer):
    """Strategia Multi-Armed Bandit (non ancora implementata)."""
    mode = LoadBalancerMode.MAB


_UNAVAILABLE_LOAD_BALANCERS: dict[LoadBalancerMode, type[_UnavailableLoadBalancer]] = {
    LoadBalancerMode.MAB: MABLoadBalancer,
}
