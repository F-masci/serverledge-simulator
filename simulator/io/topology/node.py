"""Configurazione di un singolo nodo nello scenario.

Esempio TOML nello scenario (dentro un ring):

    [[regions.edge.infrastructure.rings.nodes]]
    name = "edge-1"                       # obbligatorio (univoco nella regione)
    machine_tag = "amd64"                 # opzionale (default: machine_tag del ring)
    policy = "edgecloud"                  # opzionale (default: scheduler.policy)
    fallback_local = true                 # opzionale (default: scheduler.fallback_local)
    nearest_neighbors = ["edge-2"]        # opzionale (max 2 altri nodi della stessa regione)
    remote_region = "cloud"               # opzionale (regione remota censita in regions)
"""

from collections.abc import Mapping
from dataclasses import dataclass

from simulator.model.scheduling.offloading import SchedulingPolicy
from simulator.utils.validation import nonempty_string


@dataclass(frozen=True)
class _NodeConfig:
    """Identità del nodo; architettura e capacità sono definite dal ring."""

    name: str
    machine_tag: str | None
    policy: SchedulingPolicy | None
    fallback_local: bool | None
    nearest_neighbors: tuple[str, ...] | None
    remote_region: str | None

    @classmethod
    def parse(cls, data: object) -> '_NodeConfig':
        """Parse della configurazione di un nodo."""
        required = {'name'}
        optional = {'machine_tag', 'policy', 'remote_region', 'fallback_local', 'nearest_neighbors'}
        if not isinstance(data, Mapping) or required - data.keys() or data.keys() - required - optional:
            raise ValueError(f'Ogni nodo richiede name. Campi opzionali: {", ".join(sorted(optional))}')
        name = nonempty_string(data['name'], 'node.name')
        tag = nonempty_string(data['machine_tag'], 'node.machine_tag') if 'machine_tag' in data else None

        policy = SchedulingPolicy(nonempty_string(data['policy'], 'node.policy')) if 'policy' in data else None
        region = nonempty_string(data['remote_region'], 'node.remote_region') if 'remote_region' in data else None
        fallback: bool | None = None
        if 'fallback_local' in data:
            if not isinstance(data['fallback_local'], bool):
                raise ValueError('node.fallback_local deve essere un booleano')
            fallback = data['fallback_local']

        neighbors = cls._parse_neighbors(data['nearest_neighbors'], name) if 'nearest_neighbors' in data else None
        return cls(name, tag, policy, fallback, neighbors, region)

    @staticmethod
    def _parse_neighbors(raw_neighbors: object, node_name: str) -> tuple[str, ...]:
        """Valida l'elenco di al più due vicini edge distinti della stessa regione."""
        if not isinstance(raw_neighbors, list):
            raise ValueError('node.nearest_neighbors deve essere una lista di nomi')
        neighbors = tuple(nonempty_string(item, 'node.nearest_neighbors') for item in raw_neighbors)
        if len(neighbors) != len(set(neighbors)):
            raise ValueError('node.nearest_neighbors contiene nomi duplicati')
        if len(neighbors) > 2 or node_name in neighbors:
            raise ValueError('node.nearest_neighbors ammette fino a due altri nodi della stessa regione')
        return neighbors
