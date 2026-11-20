"""Anello dei nodi di una singola architettura."""

from bisect import bisect_left
from collections.abc import Iterator
from itertools import chain

from sim.utils.validation import nonempty_string, positive_integer, positive_number
from .node import Node
from .request import Request
from .function import Function


class ArchitectureRing:
    """Possiede i riferimenti ai nodi in ordine di inserimento.

    Ogni nodo eredita architettura e capacità CPU/RAM dall'anello.
    La selezione usa consistent hashing con repliche virtuali.
    """

    def __init__(self, ring_id: str, arch: str, memory: int, cpus: float, replicas: int = 128) -> None:
        self._id = nonempty_string(ring_id, 'ring_id')
        self._arch = nonempty_string(arch, 'arch')
        self._memory = positive_integer(memory, 'memory')
        self._cpus = positive_number(cpus, 'cpus')
        self._nodes: list[Node] = []
        self._replicas = positive_integer(replicas, 'replicas')
        self._positions: list[int] = []
        self._targets: dict[int, Node] = {}

    @staticmethod
    def _hash(key: str) -> int:
        """FNV-1a sui byte UTF-8, con overflow unsigned a 32 bit."""
        value = 2166136261
        for byte in key.encode('utf-8'):
            value = ((value ^ byte) * 16777619) & 0xFFFFFFFF
        return value

    @property
    def id(self) -> str:
        return self._id

    @property
    def memory(self) -> int:
        """Capacità RAM comune a tutti i nodi, in unità da 2**20 byte."""
        return self._memory

    @property
    def cpus(self) -> float:
        """Capacità CPU comune, distinta dalle prenotazioni dei singoli nodi."""
        return self._cpus

    @property
    def arch(self) -> str:
        """Identificatore dell'architettura comune a tutti i nodi."""
        return self._arch

    @property
    def nodes(self) -> tuple[Node, ...]:
        """Snapshot dei riferimenti, senza esporre la lista interna modificabile."""
        return tuple(self._nodes)

    @property
    def replicas(self) -> int:
        """Numero di repliche virtuali per nodo, usate per il consistent hashing."""
        return self._replicas

    def create_node(
        self, name: str, area: str,
        machine_tag: str | None = None,
        *, idle_timeout: float = 600, queue_capacity: int = 0,
    ) -> Node:
        """Costruisce e registra un nodo dell'architettura dell'anello."""

        name = nonempty_string(name, 'name')
        nonempty_string(area, 'area')

        if any(node.id.key == name for node in self._nodes):
            raise ValueError(f'Nodo già presente nell\'anello: {name}')
        
        node = Node(name, self.memory, self.cpus, area, self.arch, machine_tag, idle_timeout=idle_timeout, queue_capacity=queue_capacity)
        for index in range(self._replicas):
            position = self._hash(f'{node.id.key}#{index}')
            self._positions.append(position)
            self._targets[position] = node

        self._positions.sort()
        self._nodes.append(node)
        return node

    def _candidates(self, function: Function) -> Iterator[Node]:
        """Visita i nodi fisici una sola volta, a partire dal successore dell'hash."""
        size = len(self._positions)
        start = bisect_left(self._positions, self._hash(function.name))
        seen: set[Node] = set()
        for offset in range(size):
            node = self._targets[self._positions[(start + offset) % size]]
            if node in seen:
                continue
            seen.add(node)
            yield node
            if len(seen) == len(self._nodes):
                break

    def select_node(self, function: Function) -> Node | None:
        """Restituisce il primo candidato del consistent hashing."""
        return next(self._candidates(function), None)

    def submit(self, request: Request) -> Node | None:
        """Tenta l'ammissione in ordine di hash fino al primo nodo disponibile."""
        request.ensure_created()
        candidates = self._candidates(request.function)
        primary = next(candidates, None)
        if primary is None:
            request.reject('no_destination')
            return None

        for node in chain((primary,), candidates):
            if node.submit(request, reject_on_failure=False):
                return node

        return primary if primary.enqueue_or_reject(request) else None

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __contains__(self, node: object) -> bool:
        return node in self._nodes
