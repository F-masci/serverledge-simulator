"""Anello dei nodi di una singola architettura."""

import json
from bisect import bisect_left
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from simulator.utils.validation import nonempty_string, positive_integer, positive_number

from .node import Node
from ..execution.function import Function
from ..execution.request import Request

if TYPE_CHECKING:
    from numpy.random import Generator
    from simulator.bindings import NodeBindings
    from .region import Region
    from .infrastructure import Infrastructure


class ArchitectureRing:
    """Possiede i riferimenti ai nodi di un singolo machine_tag in ordine di inserimento.

    Ogni nodo eredita architettura, machine_tag e capacità CPU/RAM dall'anello.
    La selezione usa consistent hashing con repliche virtuali.
    """

    def __init__(
        self,
        ring_id: str,
        arch: str,
        memory: int,
        cpus: float,
        infrastructure: 'Infrastructure',
        replicas: int = 128,
        *,
        machine_tag: str | None = None,
        node_bindings: 'NodeBindings | None' = None,
    ) -> None:
        # Identificatore univoco dell'anello
        self._id = nonempty_string(ring_id, 'ring_id')

        # Capacità e machine_tag comuni a tutti i nodi dell'anello (1 ring = 1 machine_tag come in Go)
        self._arch = nonempty_string(arch, 'arch')
        self._machine_tag = self._arch if not machine_tag else nonempty_string(machine_tag, 'machine_tag')
        self._memory = positive_integer(memory, 'memory')
        self._cpus = positive_number(cpus, 'cpus')

        # Riferimento all'infrastruttura che possiede l'anello e binding dei nodi
        self._infrastructure = infrastructure
        self._node_bindings = node_bindings
        # Lista dei nodi fisici registrati nell'anello
        self._nodes: list[Node] = []
        self._replicas = positive_integer(replicas, 'replicas')
        self._positions: list[int] = []
        self._targets: dict[int, list[Node]] = {}

        ### Callback ###
        self.on_node_created: Callable[[ArchitectureRing, Node], None] | None = None
        self.on_node_removed: Callable[[ArchitectureRing, Node], None] | None = None

    ### Identification ###

    @property
    def id(self) -> str:
        return self._id

    @property
    def arch(self) -> str:
        """Restituisce l'architettura comune ai nodi dell'anello."""
        return self._arch

    @property
    def machine_tag(self) -> str:
        """Restituisce il machine_tag comune ai nodi dell'anello."""
        return self._machine_tag

    ### Infrastructure ###

    @property
    def infrastructure(self) -> 'Infrastructure':
        assert self._infrastructure is not None, 'Anello non associato a un\'infrastruttura'
        return self._infrastructure

    @property
    def region(self) -> 'Region':
        return self.infrastructure.region

    @staticmethod
    def _hash(key: str) -> int:
        """Calcola l'hash FNV-1a sui byte UTF-8 con aritmetica a 32 bit senza segno."""
        value = 2166136261
        for byte in key.encode('utf-8'):
            value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
        return value

    #################################
    ### Internal state management ###
    #################################

    ## Infra management ##

    def create_node(
        self, name: str, machine_tag: str | None = None,
        *, idle_timeout: float = 600, queue_capacity: int = 0,
        eviction_rng: 'Generator | None' = None,
    ) -> Node:
        """Crea un nodo con le capacità, l'architettura e il machine_tag dell'anello."""
        name = nonempty_string(name, 'name')

        if any(node.key == name for node in self._nodes):
            raise ValueError(f'Nodo già presente nell\'anello: {name}')

        # Tutti i nodi dell'anello devono condividere lo stesso machine_tag (1 ring = 1 machine_tag).
        effective_tag = self._machine_tag if not machine_tag else nonempty_string(machine_tag, 'machine_tag')
        if effective_tag != self._machine_tag:
            raise ValueError(
                f"machine_tag del nodo ({effective_tag}) incompatibile con quello dell'anello ({self._machine_tag})"
            )

        node = Node(
            name, self._memory, self._cpus, self._arch, self._machine_tag,
            idle_timeout=idle_timeout, queue_capacity=queue_capacity, ring=self,
            eviction_rng=eviction_rng,
        )

        # Registra le repliche virtuali e il nodo prima di applicare i binding e notificare.
        self._register_replicas(node)
        self._positions.sort()
        self._nodes.append(node)

        # Applica i binding di ciclo di vita al livello effettivo di creazione del nodo.
        if self._node_bindings is not None:
            self._node_bindings.bind(node)

        if self.on_node_created is not None:
            self.on_node_created(self, node)

        return node

    def remove_node(self, name: str) -> Node:
        """Rimuove un nodo per nome dal routing e restituisce il nodo rimosso."""
        name = nonempty_string(name, 'name')
        node = next((node for node in self._nodes if node.key == name), None)
        if node is None:
            raise ValueError(f'Nodo non trovato nell\'anello: {name}')

        self._nodes.remove(node)

        # Mantiene le posizioni condivise finché contengono almeno una replica.
        for position in self._replica_positions(node):
            targets = self._targets[position]
            targets.remove(node)
            if not targets:
                del self._targets[position]
                self._positions.pop(bisect_left(self._positions, position))

        # Mantiene node._ring integro per consentire il drain delle invocazioni e dei
        # container già in volo; il nodo è comunque escluso dal routing (node not in self).
        if self._node_bindings is not None and self._node_bindings.on_node_removed is not None:
            self._node_bindings.on_node_removed(node)
        if self.on_node_removed is not None:
            self.on_node_removed(self, node)

        return node

    def select_node(self, function: Function) -> Node | None:
        """Restituisce il primo candidato del consistent hashing."""
        return next(self._candidates(function), None)

    def _candidates(self, function: Function) -> Iterator[Node]:
        """Visita i nodi fisici una sola volta, a partire dal successore dell'hash."""
        target_count = len(self._nodes)
        size = len(self._positions)
        if target_count == 0 or size == 0:
            return

        # Trova la posizione iniziale dell'hash della funzione e visita i nodi in ordine circolare.
        start = bisect_left(self._positions, self._hash(function.name))
        seen: set[Node] = set()
        for offset in range(size):
            targets = self._targets[self._positions[(start + offset) % size]]
            node = targets[-1]
            if node in seen:
                continue
            seen.add(node)
            yield node
            if len(seen) == target_count:
                break

    def _register_replicas(self, node: Node) -> None:
        """Aggiunge all'indice le repliche virtuali del nodo."""
        for position in self._replica_positions(node):
            if position not in self._targets:
                self._positions.append(position)
                self._targets[position] = []
            # In caso di collisione il routing usa la replica registrata per ultima.
            self._targets[position].append(node)

    def _replica_positions(self, node: Node) -> Iterator[int]:
        """Calcola le posizioni delle repliche virtuali del nodo."""
        for index in range(self._replicas):
            yield self._hash(f'{node.id.key}#{index}')

    ## Request management ##

    def select_eligible_node(
        self,
        function: Function,
        *,
        is_eligible: Callable[[Node], bool] | None = None,
    ) -> Node | None:
        """Seleziona il primo candidato dell'hash ring che soddisfa il criterio di ammissione."""
        predicate = (lambda candidate: candidate.can_allocate(function)) if is_eligible is None else is_eligible
        return next((candidate for candidate in self._candidates(function) if predicate(candidate)), None)

    def submit(
        self,
        request: Request,
        *,
        now: float = 0.0,
        is_eligible: Callable[[Node], bool] | None = None,
    ) -> Node | None:
        """Seleziona un nodo idoneo con l'hash e gli consegna l'invocazione."""
        request.ensure_created()
        node = self.select_eligible_node(request.function, is_eligible=is_eligible)
        if node is None:
            return None

        destination = node.receive(request, now=now)
        return destination

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile dell'anello."""
        return (
            f'ArchitectureRing(id={self._id}, arch={self._arch}, machine_tag={self._machine_tag}, '
            f'memory_mb={self._memory}, cpus={self._cpus}, nodes={len(self._nodes)})'
        )

    def to_dict(self) -> dict:
        """Restituisce la configurazione e lo stato dei nodi dell'anello in un formato serializzabile."""
        return {
            'id': self._id,
            'arch': self._arch,
            'machine_tag': self._machine_tag,
            'memory_mb': self._memory,
            'cpus': self._cpus,
            'replicas': self._replicas,
            'region': self.region.name,
            'nodes': [node.to_dict() for node in self._nodes],
        }

    def to_json(self) -> str:
        """Serializza l'anello per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<ArchitectureRing id={self._id} arch={self._arch} nodes={len(self._nodes)}>'

    ###############
    ### Utility ###
    ###############

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes)

    def __contains__(self, node: object) -> bool:
        return node in self._nodes
