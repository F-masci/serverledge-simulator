"""Anello dei nodi di una singola architettura."""

from bisect import bisect_left
from collections.abc import Iterator

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
    ) -> Node:
        """Costruisce e registra un nodo dell'architettura dell'anello."""

        name = nonempty_string(name, 'name')
        nonempty_string(area, 'area')

        if any(node.id.key == name for node in self._nodes):
            raise ValueError(f'Nodo già presente nell\'anello: {name}')
        
        node = Node(name, self.memory, self.cpus, area, self.arch, machine_tag)
        for index in range(self._replicas):
            position = self._hash(f'{node.id.key}#{index}')
            self._positions.append(position)
            self._targets[position] = node

        self._positions.sort()
        self._nodes.append(node)
        return node

    def select_node(self, function: Function) -> Node | None:
        """Cerca il successore dell'hash della funzione, con ritorno all'inizio."""
        if not self._positions:
            return None

        index = bisect_left(self._positions, self._hash(function.name))
        return self._targets[self._positions[index % len(self._positions)]]

    def submit(self, request: Request) -> Node | None:
        """Seleziona il nodo tramite hashing e gli consegna la stessa richiesta."""
        request._ensure_new()

        node = self.select_node(request.function)
        if node is None:
            request._reject('no_destination')
            return None

        return node if node.submit(request) else None

    def __len__(self) -> int:
        return len(self._nodes)

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __contains__(self, node: object) -> bool:
        return node in self._nodes
