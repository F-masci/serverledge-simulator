"""Configurazione dell'infrastruttura di una regione nello scenario.

Esempio TOML nello scenario (singola regione o dentro una regione in [regions.<nome>]):

    [[regions.edge.infrastructure.rings]]   # obbligatorio: chiave 'rings' (lista di ring)
    id = "edge-x86"
    arch = "amd64"
    memory_mb = 512
    cpus = 2.0
    nodes = [{ name = "x86-1" }]
"""

from collections.abc import Mapping
from dataclasses import dataclass

from .ring import _RingConfig


@dataclass(frozen=True)
class _InfrastructureConfig:
    """Topologia dichiarativa dalla quale ogni replica costruisce i suoi nodi."""

    rings: tuple[_RingConfig, ...]

    @classmethod
    def parse(cls, infrastructure: object) -> '_InfrastructureConfig':
        """Parse della configurazione dell'infrastruttura."""
        if infrastructure is None:
            return cls(())

        if not isinstance(infrastructure, Mapping) or infrastructure.keys() != {'rings'}:
            raise ValueError('infrastructure richiede rings')
        definitions = infrastructure['rings']
        if not isinstance(definitions, list):
            raise ValueError('infrastructure.rings deve essere una lista')

        rings: dict[str, _RingConfig] = {}
        seen_tags: set[str] = set()
        seen_nodes: set[str] = set()
        for definition in definitions:
            ring = _RingConfig.parse(definition)
            if ring.id in rings:
                raise ValueError(f'Ring già definito: {ring.id}')
            if ring.machine_tag in seen_tags:
                raise ValueError(f'Ring duplicato per il machine_tag: {ring.machine_tag}')
            for node in ring.nodes:
                if node.name in seen_nodes:
                    raise ValueError(f'Nodo già definito nella regione: {node.name}')
                seen_nodes.add(node.name)
            seen_tags.add(ring.machine_tag)
            rings[ring.id] = ring
        return cls(tuple(rings.values()))
