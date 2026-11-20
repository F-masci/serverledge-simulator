"""Strategie di selezione del ring per le invocazioni."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from enum import StrEnum
from typing import Never

from sim.random.streams import RandomStreams, StreamKind

from .ring import ArchitectureRing
from .request import Request


class LoadBalancerMode(StrEnum):
    RANDOM = 'Random'
    MAB = 'MAB'
    ROUND_ROBIN = 'RoundRobin'
    ARCHITECTURE_UNAWARE = 'ArchitectureUnaware'


class LoadBalancer(ABC):
    def submit(
        self, request: Request, rings: Iterable[ArchitectureRing],
    ) -> ArchitectureRing | None:
        """Restituisce il ring primario per l'invocazione."""
        return next(iter(self.candidate_rings(request, rings)), None)

    def candidate_rings(
        self, request: Request, rings: Iterable[ArchitectureRing],
    ) -> tuple[ArchitectureRing, ...]:
        """Restituisce il primario seguito dai ring compatibili in ordine di registrazione."""
        function = request.function
        compatible = tuple(
            ring for ring in rings
            if len(ring)
            and function.supports_arch(ring.arch)
            and any(
                function.supports_machine_tag(node.id.machine_tag)
                for node in ring
            )
        )
        if not compatible:
            return ()

        index = self._select_index(request, compatible)
        return (compatible[index], *compatible[:index], *compatible[index + 1:])

    @abstractmethod
    def _select_index(self, request: Request, rings: tuple[ArchitectureRing, ...]) -> int:
        """Sceglie l'indice del ring primario tra i candidati."""


class RandomLoadBalancer(LoadBalancer):
    def __init__(self, *, seed: int = 12345, stream_id: str = 'load_balancer') -> None:
        self._rng = RandomStreams(seed).stream(StreamKind.ROUTING, stream_id)

    def _select_index(self, request: Request, rings: tuple[ArchitectureRing, ...]) -> int:
        return int(self._rng.integers(len(rings)))


class _UnavailableLoadBalancer(LoadBalancer):
    mode: LoadBalancerMode

    def __init__(self) -> None:
        self._raise_unavailable()

    def _select_index(self, request: Request, rings: tuple[ArchitectureRing, ...]) -> int:
        self._raise_unavailable()

    def _raise_unavailable(self) -> Never:
        raise NotImplementedError(f'Load balancer {self.mode} non disponibile nel simulatore')


class MABLoadBalancer(_UnavailableLoadBalancer):
    mode = LoadBalancerMode.MAB


class RoundRobinLoadBalancer(_UnavailableLoadBalancer):
    mode = LoadBalancerMode.ROUND_ROBIN


class ArchitectureUnawareLoadBalancer(_UnavailableLoadBalancer):
    mode = LoadBalancerMode.ARCHITECTURE_UNAWARE


def build_load_balancer(mode: LoadBalancerMode, *, seed: int, stream_id: str = 'load_balancer') -> LoadBalancer:
    if mode is LoadBalancerMode.RANDOM:
        return RandomLoadBalancer(seed=seed, stream_id=stream_id)
    unavailable = {
        LoadBalancerMode.MAB: MABLoadBalancer,
        LoadBalancerMode.ROUND_ROBIN: RoundRobinLoadBalancer,
        LoadBalancerMode.ARCHITECTURE_UNAWARE: ArchitectureUnawareLoadBalancer,
    }
    return unavailable[mode]()
