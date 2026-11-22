"""Strategie di selezione dell'anello di destinazione per le invocazioni."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum
import json
from typing import ClassVar, Never

from sim.random.streams import RandomStreams, StreamKind
from .request import Request
from .ring import ArchitectureRing


class LoadBalancerMode(StrEnum):
    """Strategie di bilanciamento del carico selezionabili nello scenario."""
    RANDOM = 'Random'                             # Selezione casuale uniforme tra gli anelli compatibili.
    ROUND_ROBIN = 'RoundRobin'                    # Selezione circolare tra gli anelli compatibili.
    MAB = 'MAB'                                   # Multi-Armed Bandit (non ancora disponibile nel simulatore).
    ARCHITECTURE_UNAWARE = 'ArchitectureUnaware'  # Selezione indipendente dall'architettura (non ancora disponibile).


class LoadBalancer(ABC):
    """
    Contratto comune delle strategie di bilanciamento tra anelli di una regione.

    Filtra gli anelli compatibili con la funzione richiesta (non vuoti, architettura
    supportata e almeno un nodo compatibile con il machine tag) e determina l'ordine
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

        # Le strategie non ancora modellate sollevano NotImplementedError in fase di costruzione.
        return _UNAVAILABLE_LOAD_BALANCERS[mode]()

    #################################
    ### Internal state management ###
    #################################

    ## Ring selection ##

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

        # Filtra gli anelli che possiedono almeno un nodo compatibile
        # con l'architettura e il pattern di machine tag della funzione.
        compatible = tuple(
            ring for ring in registered
            if len(ring)
            and function.supports_arch(ring.arch)
            and any(
                function.supports_machine_tag(node.id.machine_tag)
                for node in ring
            )
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


class ArchitectureUnawareLoadBalancer(_UnavailableLoadBalancer):
    """Strategia indipendente dall'architettura (non ancora implementata)."""
    mode = LoadBalancerMode.ARCHITECTURE_UNAWARE


_UNAVAILABLE_LOAD_BALANCERS: dict[LoadBalancerMode, type[_UnavailableLoadBalancer]] = {
    LoadBalancerMode.MAB: MABLoadBalancer,
    LoadBalancerMode.ARCHITECTURE_UNAWARE: ArchitectureUnawareLoadBalancer,
}
