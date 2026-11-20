"""Infrastruttura organizzata in anelli omogenei identificati esplicitamente."""

from collections.abc import Mapping
from types import MappingProxyType

from sim.utils.validation import nonempty_string
from .ring import ArchitectureRing
from .node import Node
from .request import Request
from .load_balancer import LoadBalancer


class Infrastructure:
    def __init__(self, load_balancer: LoadBalancer | None = None) -> None:
        """Inizializza un'infrastruttura senza nodi né anelli."""
        self._rings: dict[str, ArchitectureRing] = {}
        self._load_balancer = LoadBalancer() if load_balancer is None else load_balancer

    @property
    def rings(self) -> Mapping[str, ArchitectureRing]:
        """Vista non modificabile dell'associazione ID → anello."""
        return MappingProxyType(self._rings)

    @property
    def load_balancer(self) -> LoadBalancer:
        return self._load_balancer

    def get_ring(self, ring_id: str) -> ArchitectureRing | None:
        """Restituisce l'anello identificato dall'ID, o None se non esiste."""
        return self._rings.get(ring_id)

    def create_ring(
        self,
        ring_id: str,
        *,
        arch: str | None = None,
        memory: int | None = None,
        cpus: float | None = None,
        replicas: int | None = None,
    ) -> ArchitectureRing:
        """Crea un anello configurato o restituisce quello identificato dall'ID."""
        ring_id = nonempty_string(ring_id, 'ring_id')

        # Se l'anello esiste già, restituiscilo senza modificare le capacità.
        ring = self.get_ring(ring_id)
        if ring is not None:
            return ring

        # Se l'anello non esiste, richiede architettura, CPU e memoria per crearlo.
        if arch is None or memory is None or cpus is None:
            raise ValueError('Un nuovo anello richiede architettura, CPU e memoria')
        replicas = 128 if replicas is None else replicas
        ring = ArchitectureRing(ring_id, arch, memory, cpus, replicas)
        self._rings[ring_id] = ring
        return ring


    def create_node(
        self,
        ring_id: str,
        name: str,
        area: str,
        machine_tag: str | None = None,
        *,
        idle_timeout: float = 600,
        queue_capacity: int = 0,
    ) -> Node:
        """Crea un nodo nell'anello specificato, creando l'anello se necessario."""
        ring = self.get_ring(ring_id)
        if ring is None:
            raise ValueError(f'Anello non trovato: {ring_id}')
        return ring.create_node(name, area, machine_tag, idle_timeout=idle_timeout, queue_capacity=queue_capacity)

    def submit(self, request: Request) -> Node | None:
        """Inoltra una richiesta all'infrastruttura."""
        request.ensure_created()
        ring = self._load_balancer.submit(request, self._rings.values())
        if ring is None:
            request.reject('no_destination')
            return None
        return ring.submit(request)
