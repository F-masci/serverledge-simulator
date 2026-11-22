"""Infrastruttura organizzata in anelli omogenei identificati esplicitamente."""

from collections.abc import Callable, Mapping
import json
from types import MappingProxyType
from typing import TYPE_CHECKING

from .load_balancer import LoadBalancer, RandomLoadBalancer
from .node import Node
from .offloading import NodeScheduling, OffloadingScheduler
from .request import Request
from .ring import ArchitectureRing
from ..utils.validation import nonempty_string

if TYPE_CHECKING:
    from numpy.random import Generator
    from sim.bindings import NodeBindings
    from .region import Region


class Infrastructure:
    """Raggruppa gli anelli e instrada le richieste verso i nodi."""

    def __init__(
        self, load_balancer: LoadBalancer | None = None, *,
        region: 'Region | None' = None,
        node_bindings: 'NodeBindings | None' = None,
    ) -> None:
        self._rings: dict[str, ArchitectureRing] = {}
        self._region = region
        self._load_balancer = RandomLoadBalancer() if load_balancer is None else load_balancer
        self._node_bindings = node_bindings

        ### Callback ###
        self.on_ring_created: Callable[[Infrastructure, ArchitectureRing], None] | None = None
        self.on_node_created: Callable[[Infrastructure, ArchitectureRing, Node], None] | None = None
        self.on_node_removed: Callable[[Infrastructure, ArchitectureRing, Node], None] | None = None
        self.on_request_submitted: Callable[[Infrastructure, Request, Node | None], None] | None = None

    @property
    def region(self) -> Region:
        assert self._region is not None, 'Infrastruttura non associata a una regione'
        return self._region

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
        replicas: int | None = None, machine_tag: str | None = None,
    ) -> ArchitectureRing:
        """Crea un anello configurato o restituisce quello identificato dall'ID."""
        ring_id = nonempty_string(ring_id, 'ring_id')

        # Riutilizza l'anello esistente senza modificare le capacità.
        ring = self.get_ring(ring_id)
        if ring is not None:
            return ring

        if arch is None or memory is None or cpus is None:
            raise ValueError('Un nuovo anello richiede architettura, CPU e memoria')
        resolved_tag = arch if not machine_tag else nonempty_string(machine_tag, 'machine_tag')
        if any(existing.machine_tag == resolved_tag for existing in self._rings.values()):
            raise ValueError(f'Esiste già un anello per il machine_tag: {resolved_tag}')

        replicas = 128 if replicas is None else replicas
        ring = ArchitectureRing(
            ring_id, arch, memory, cpus, infrastructure=self, replicas=replicas,
            machine_tag=resolved_tag, node_bindings=self._node_bindings,
        )

        # Registra l'anello prima di notificare la creazione.
        self._rings[ring_id] = ring
        if self.on_ring_created is not None:
            self.on_ring_created(self, ring)
        return ring

    def create_node(
        self, ring_id: str, name: str, machine_tag: str | None = None,
        *, idle_timeout: float = 600, queue_capacity: int = 0,
        eviction_rng: 'Generator | None' = None,
    ) -> Node:
        """Crea un nodo nell'anello specificato, che deve già esistere."""
        ring = self.get_ring(ring_id)
        if ring is None:
            raise ValueError(f'Anello non trovato: {ring_id}')

        # L'anello registra il nodo prima della notifica.
        node = ring.create_node(
            name, machine_tag,
            idle_timeout=idle_timeout, queue_capacity=queue_capacity, eviction_rng=eviction_rng,
        )
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

    def resolve_node(self, name: str) -> Node:
        matches = [node for node in self._nodes() if node.id.key == name]
        if len(matches) != 1:
            raise ValueError(f'Nodo di scenario assente o ambiguo: {name}')
        return matches[0]

    def configure_scheduler(self, node: Node, settings: NodeScheduling) -> None:
        """Configura la policy del nodo con la vista dei vicini della regione."""
        node.scheduler = OffloadingScheduler.create(self._nodes, settings)

    def _nodes(self) -> tuple[Node, ...]:
        return tuple(node for ring in self._rings.values() for node in ring)

    def submit(
        self, request: Request, *, now: float = 0.0, node: Node | None = None,
    ) -> Node | None:
        """Riceve un'invocazione al bilanciatore oppure direttamente a un nodo."""
        request.ensure_created()
        if node is None:
            destination = self._route(request, now=now)
        else:
            destination = self._submit_to_node(request, node, now=now)

        if self.on_request_submitted is not None:
            self.on_request_submitted(self, request, destination)
        return destination

    def _route(self, request: Request, *, now: float = 0.0) -> Node | None:
        return self._load_balancer.route(request, self._rings.values(), now=now)

    def _submit_to_node(self, request: Request, node: Node, *, now: float = 0.0) -> Node | None:
        ring = node.ring
        if ring is None or ring.infrastructure is not self or node not in ring:
            request.reject('no_destination')
            return None
        return node.receive(request, now=now)

    def forward(self, request: Request, target: Node, *, source: Node, now: float = 0.0) -> Node | None:
        """Consegna l'invocazione al nodo vicino scelto dalla policy di offloading."""
        # TODO: modellare la latenza di rete edge-to-edge (coordinate Vivaldi) per l'offloading intra-regione.
        request.mark_offloaded(source)
        return self._submit_to_node(request, target, now=now)

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile dell'infrastruttura."""
        region = self._region.name if self._region is not None else None
        return f'Infrastructure(region={region}, rings={len(self._rings)})'

    def to_dict(self) -> dict:
        """Restituisce la topologia e lo stato dell'infrastruttura in formato serializzabile."""
        return {
            'region': self._region.name if self._region is not None else None,
            'load_balancer': self._load_balancer.to_dict(),
            'rings': {ring_id: ring.to_dict() for ring_id, ring in self._rings.items()},
        }

    def to_json(self) -> str:
        """Serializza l'infrastruttura per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        region = self._region.name if self._region is not None else None
        return f'<Infrastructure region={region} rings={len(self._rings)}>'
