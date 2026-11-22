"""Strategie di selezione dell'anello di destinazione per le invocazioni."""

from abc import ABC, abstractmethod
from bisect import bisect_left
from collections.abc import Iterable, Iterator
from enum import StrEnum
import json
from typing import ClassVar, Never

from sim.random.streams import RandomStreams, StreamKind
from sim.utils.validation import positive_integer
from .function import Function
from .node import Node
from .request import Request, RequestStatus
from .ring import ArchitectureRing


class LoadBalancerMode(StrEnum):
    """Strategie di bilanciamento del carico selezionabili nello scenario."""
    RANDOM = 'Random'                             # Selezione casuale uniforme tra gli anelli compatibili.
    ROUND_ROBIN = 'RoundRobin'                    # Selezione circolare tra gli anelli compatibili.
    MAB = 'MAB'                                   # Multi-Armed Bandit (non ancora disponibile nel simulatore).
    ARCHITECTURE_UNAWARE = 'ArchitectureUnaware'  # Singolo hash ring globale indipendente da architettura e machine_tag.


class LoadBalancer(ABC):
    """
    Contratto comune delle strategie di bilanciamento tra anelli di una regione.

    Filtra gli anelli compatibili con la funzione richiesta (non vuoti, architettura
    supportata e compatibilità con il machine tag) e determina l'ordine
    con cui l'infrastruttura tenta la consegna dell'invocazione.
    """

    mode: ClassVar[LoadBalancerMode]

    @staticmethod
    def create(mode: LoadBalancerMode, *, seed: int, stream_id: str = 'load_balancer') -> 'LoadBalancer':
        """Costruisce il load balancer corrispondente alla modalità configurata nello scenario."""
        if mode is LoadBalancerMode.RANDOM:
            return RandomLoadBalancer(seed=seed, stream_id=stream_id)
        if mode is LoadBalancerMode.ROUND_ROBIN:
            return RoundRobinLoadBalancer()
        if mode is LoadBalancerMode.ARCHITECTURE_UNAWARE:
            return ArchitectureUnawareLoadBalancer()

        # Le strategie non ancora modellate sollevano NotImplementedError in fase di costruzione.
        return _UNAVAILABLE_LOAD_BALANCERS[mode]()

    #################################
    ### Internal state management ###
    #################################

    ## Ring selection ##

    def route(
        self, request: Request, rings: Iterable[ArchitectureRing], *, now: float = 0.0,
    ) -> Node | None:
        """Instrada l'invocazione sugli anelli compatibili e gestisce il fallback."""
        candidates = self.candidate_rings(request, rings)

        # Privilegia l'esecuzione immediata su qualsiasi ring compatibile.
        for ring in candidates:
            destination = ring.submit(request, now=now)
            if request.status is not RequestStatus.CREATED:
                # La risposta dello scheduler arriva dopo la selezione del proxy.
                return destination

        # A saturazione, consente accodamento FIFO o offloading sui nodi idonei.
        for ring in candidates:
            destination = ring.submit(request, now=now, allow_saturated=True)
            if request.status is not RequestStatus.CREATED:
                return destination

        request.reject('out_of_resources' if candidates else 'no_destination')
        return None

    def submit(
        self, request: Request, rings: Iterable[ArchitectureRing],
    ) -> ArchitectureRing | None:
        """Restituisce l'anello primario selezionato per l'invocazione, o None se assente."""
        return next(iter(self.candidate_rings(request, rings)), None)

    def candidate_rings(
        self, request: Request, rings: Iterable[ArchitectureRing],
    ) -> tuple[ArchitectureRing, ...]:
        """Restituisce l'anello primario seguito dagli altri anelli compatibili in ordine di registrazione."""
        function = request.function
        registered = tuple(rings)

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
        }

    def to_json(self) -> str:
        """Serializza il load balancer per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} mode={self.mode.value}>'


class RandomLoadBalancer(LoadBalancer):
    """Seleziona casualmente l'anello primario tra quelli compatibili."""

    mode = LoadBalancerMode.RANDOM

    def __init__(self, *, seed: int = 12345, stream_id: str = 'load_balancer') -> None:
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

    def __init__(self) -> None:
        # Indice circolare sull'elenco degli anelli registrati nella regione.
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

    def __init__(self, *, replicas: int = 128) -> None:
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

        # TODO: modellare NodeMetricCache del proxy come in ArchitectureUnawareBalancer.Next.
        node = next((c for c in candidates if c.can_allocate(function)), None)
        if node is None:
            node = next(
                (c for c in candidates if c.can_enqueue(request) or c.can_offload_request(request)),
                None,
            )
        if node is None:
            request.reject('out_of_resources')
            return None

        return node.receive(request, now=now)


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
