"""Infrastruttura organizzata in anelli omogenei identificati esplicitamente."""

from collections.abc import Callable, Mapping
from types import MappingProxyType

from .load_balancer import LoadBalancer
from .node import Node
from .request import Request
from .ring import ArchitectureRing
from ..utils.validation import nonempty_string


class Infrastructure:
    """Raggruppa gli anelli e instrada le richieste verso i nodi."""

    def __init__(self, load_balancer: LoadBalancer | None = None) -> None:
        self._rings: dict[str, ArchitectureRing] = {}
        self._load_balancer = LoadBalancer() if load_balancer is None else load_balancer

        ### Callback ###
        self.on_ring_created: Callable[[Infrastructure, ArchitectureRing], None] | None = None
        self.on_node_created: Callable[[Infrastructure, ArchitectureRing, Node], None] | None = None
        self.on_node_removed: Callable[[Infrastructure, ArchitectureRing, Node], None] | None = None
        self.on_request_submitted: Callable[[Infrastructure, Request, Node | None], None] | None = None

    @property
    def rings(self) -> Mapping[str, ArchitectureRing]:
        """Vista non modificabile dell'associazione ID → anello."""
        return MappingProxyType(self._rings)

    def get_ring(self, ring_id: str) -> ArchitectureRing | None:
        """Restituisce l'anello identificato dall'ID, o None se non esiste."""
        return self._rings.get(ring_id)

    def create_ring(
        self, ring_id: str, *,
        arch: str | None = None, memory: int | None = None, cpus: float | None = None,
        replicas: int | None = None,
    ) -> ArchitectureRing:
        """Crea un anello configurato o restituisce quello identificato dall'ID."""
        ring_id = nonempty_string(ring_id, 'ring_id')

        # Riutilizza l'anello esistente senza modificare le capacità.
        ring = self.get_ring(ring_id)
        if ring is not None:
            return ring

        if arch is None or memory is None or cpus is None:
            raise ValueError('Un nuovo anello richiede architettura, CPU e memoria')
        replicas = 128 if replicas is None else replicas
        ring = ArchitectureRing(ring_id, arch, memory, cpus, replicas)

        # Registra l'anello prima di notificare la creazione.
        self._rings[ring_id] = ring
        if self.on_ring_created is not None:
            self.on_ring_created(self, ring)
        return ring

    def create_node(
        self, ring_id: str, name: str, area: str, machine_tag: str | None = None,
        *, idle_timeout: float = 600, queue_capacity: int = 0,
    ) -> Node:
        """Crea un nodo nell'anello specificato, che deve già esistere."""
        ring = self.get_ring(ring_id)
        if ring is None:
            raise ValueError(f'Anello non trovato: {ring_id}')

        # L'anello registra il nodo prima della notifica.
        node = ring.create_node(name, area, machine_tag, idle_timeout=idle_timeout, queue_capacity=queue_capacity)
        if self.on_node_created is not None:
            self.on_node_created(self, ring, node)
        return node

    def remove_node(self, ring_id: str, name: str) -> Node:
        """Rimuove il nodo dall'anello specificato e restituisce il nodo rimosso."""
        ring = self.get_ring(ring_id)
        if ring is None:
            raise ValueError(f'Anello non trovato: {ring_id}')

        # L'anello aggiorna il routing prima della notifica.
        node = ring.remove_node(name)
        if self.on_node_removed is not None:
            self.on_node_removed(self, ring, node)
        return node

    def submit(self, request: Request) -> Node | None:
        """Instrada la richiesta e notifica l'esito dell'inoltro."""
        request.ensure_created()
        ring = self._load_balancer.submit(request, self._rings.values())
        if ring is None:
            request.reject('no_destination')
            node = None
        else:
            node = ring.submit(request)

        # La richiesta è già accettata, in coda o rifiutata al momento della notifica.
        if self.on_request_submitted is not None:
            self.on_request_submitted(self, request, node)
        return node
