"""Infrastruttura organizzata in anelli omogenei identificati esplicitamente."""

from collections.abc import Mapping
from types import MappingProxyType

from sim.validation import nonempty_string
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

    def create_ring(
        self, ring_id: str, *, arch: str | None = None,
        memory: int | None = None, cpus: float | None = None, replicas: int | None = None,
    ) -> ArchitectureRing:
        """Crea un anello configurato o restituisce quello identificato dall'ID.

        Un nuovo ID richiede architettura e capacità. Per un ID esistente
        eventuali parametri devono coincidere: non si riconfigura il ring.
        """
        ring_id = nonempty_string(ring_id, 'ring_id')
        if ring_id in self._rings:
            ring = self._rings[ring_id]
            for field, value in (('arch', arch), ('memory', memory), ('cpus', cpus), ('replicas', replicas)):
                if value is not None and value != getattr(ring, field):
                    raise ValueError(f'Configurazione diversa per l\'anello {ring_id}: {field}')
            return ring
        if arch is None or memory is None or cpus is None:
            raise ValueError('Un nuovo anello richiede architettura, CPU e memoria')
        ring = ArchitectureRing(ring_id, arch, memory, cpus, 128 if replicas is None else replicas)
        self._rings[ring_id] = ring
        return ring

    def create_node(
        self, ring_id: str, name: str, area: str, machine_tag: str | None = None,
    ) -> Node:
        """Ottiene il ring tramite create_ring e gli delega la creazione del nodo.

        Il ring deve essere già configurato: le capacità non sono parametri
        del singolo nodo e non possono essere sovrascritte alla creazione.
        """
        ring = self.create_ring(ring_id)
        return ring.create_node(name, area, machine_tag)

    def submit(self, request: Request) -> Node | None:
        """Chiede al bilanciatore il ring e delega a esso l'inoltro al nodo.

        None indica l'assenza di una destinazione; non viene creata una coda
        nell'infrastruttura e non vengono modificati i pool dei container.
        """
        request._ensure_new()
        ring = self._load_balancer.submit(request, self._rings.values())
        if ring is None:
            request._reject('no_destination')
            return None
        return ring.submit(request)
