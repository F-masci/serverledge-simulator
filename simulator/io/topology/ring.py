"""Configurazione di un ring omogeneo di nodi nello scenario.

Esempio TOML nello scenario:

    [[regions.edge.infrastructure.rings]]
    id = "edge-x86"              # obbligatorio (univoco nella regione)
    arch = "amd64"               # obbligatorio
    memory_mb = 512              # obbligatorio (intero > 0, in MiB)
    cpus = 2.0                   # obbligatorio (numero > 0)
    machine_tag = "amd64/std"    # opzionale (default: arch; 1 ring = 1 machine_tag)
    replicas = 128               # opzionale (default: load_balancer_replicas o 128)
    nodes = [{ name = "x86-1" }] # obbligatorio (lista anche vuota di nodi)

    # Opzionale: parametri di scaling del ring (vedi simulator/io/scaler/ring.py)
    [regions.edge.infrastructure.rings.scaling]
    min_nodes = 1
    max_nodes = 4
"""

from collections.abc import Mapping
from dataclasses import dataclass

from simulator.utils.validation import nonempty_string, positive_integer, positive_number

from .node import _NodeConfig
from ..scaler import RingScalingConfig


@dataclass(frozen=True)
class _RingConfig:
    """Capacità comuni, machine_tag e nodi appartenenti a un anello omogeneo."""

    id: str
    arch: str
    machine_tag: str
    memory_mb: int
    cpus: float
    replicas: int | None
    nodes: tuple[_NodeConfig, ...]
    scaling: RingScalingConfig | None = None

    @classmethod
    def parse(cls, data: object) -> '_RingConfig':
        """Parse della configurazione di un ring."""
        required = {'id', 'arch', 'memory_mb', 'cpus', 'nodes'}
        optional = {'replicas', 'machine_tag', 'scaling'}
        if not isinstance(data, Mapping) or required - data.keys() or data.keys() - required - optional:
            raise ValueError(
                'Ogni ring richiede id, arch, memory_mb, cpus e nodes; machine_tag, replicas e scaling sono opzionali'
            )

        ring_id = nonempty_string(data['id'], 'ring.id')
        arch = nonempty_string(data['arch'], 'ring.arch')
        ring_tag = nonempty_string(data['machine_tag'], 'ring.machine_tag') if 'machine_tag' in data else arch
        memory_mb = positive_integer(data['memory_mb'], 'ring.memory_mb')
        cpus = positive_number(data['cpus'], 'ring.cpus')
        replicas = positive_integer(data['replicas'], 'ring.replicas') if 'replicas' in data else None
        scaling = (
            RingScalingConfig.parse(data['scaling'], prefix=f'ring.{ring_id}.scaling')
            if 'scaling' in data and data['scaling'] is not None
            else None
        )

        definitions = data['nodes']
        if not isinstance(definitions, list):
            raise ValueError(f'I nodi del ring {ring_id} devono essere una lista')

        nodes: dict[str, _NodeConfig] = {}
        for definition in definitions:
            node = _NodeConfig.parse(definition)
            if node.name in nodes:
                raise ValueError(f'Nodo già definito nel ring {ring_id}: {node.name}')
            if node.machine_tag is not None and node.machine_tag != ring_tag:
                raise ValueError(
                    f"machine_tag del nodo {node.name} ({node.machine_tag}) incompatibile con il ring {ring_id} ({ring_tag})"
                )
            nodes[node.name] = node

        return cls(ring_id, arch, ring_tag, memory_mb, cpus, replicas, tuple(nodes.values()), scaling)
