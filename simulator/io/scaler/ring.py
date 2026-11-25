"""Parametri di capacità acquisibile, ritardi di transizione e potenza per singolo ring/pool.

Esempio TOML nello scenario (tutti i parametri sono opzionali):

    [regions.edge.infrastructure.rings.scaling]
    min_nodes = 1                # opzionale (default: 0)
    max_nodes = 4                # opzionale (default: max(nodi_iniziali, 1))
    boot_time_s = 5.0            # opzionale (default: 5.0 s)
    shutdown_time_s = 0.0        # opzionale (default: 0.0 s)
    provision_failure_prob = 0.0 # opzionale (default: 0.0, in [0, 1))
    idle_power_w = 50.0          # opzionale (default: 50.0 W)
    max_power_w = 150.0          # opzionale (default: 150.0 W, >= idle_power_w)
    boot_power_w = 80.0          # opzionale (default: idle_power_w)
    drain_power_w = 50.0         # opzionale (default: idle_power_w)
    allocation_cost_per_s = 0.0  # opzionale (default: 0.0)
    cooldown_s = 0.0             # opzionale (default: 0.0 s)
"""

from collections.abc import Mapping
from dataclasses import dataclass

from simulator.utils.validation import nonnegative_integer, nonnegative_number, positive_integer


@dataclass(frozen=True)
class RingScalingConfig:
    """Parametri di capacità acquisibile, ritardi di transizione e profilo di potenza di un ring."""

    min_nodes: int = 0
    max_nodes: int | None = None
    boot_time_s: float = 5.0
    shutdown_time_s: float = 0.0
    provision_failure_prob: float = 0.0
    idle_power_w: float = 50.0
    max_power_w: float = 150.0
    boot_power_w: float | None = None
    drain_power_w: float | None = None
    allocation_cost_per_s: float = 0.0
    cooldown_s: float = 0.0

    def __post_init__(self) -> None:
        nonnegative_integer(self.min_nodes, 'ring.scaling.min_nodes')
        if self.max_nodes is not None:
            positive_integer(self.max_nodes, 'ring.scaling.max_nodes')
            if self.min_nodes > self.max_nodes:
                raise ValueError('ring.scaling.min_nodes non può superare max_nodes')
        nonnegative_number(self.boot_time_s, 'ring.scaling.boot_time_s')
        nonnegative_number(self.shutdown_time_s, 'ring.scaling.shutdown_time_s')
        prob = nonnegative_number(self.provision_failure_prob, 'ring.scaling.provision_failure_prob')
        if prob >= 1.0:
            raise ValueError('ring.scaling.provision_failure_prob deve essere in [0, 1)')
        idle_w = nonnegative_number(self.idle_power_w, 'ring.scaling.idle_power_w')
        max_w = nonnegative_number(self.max_power_w, 'ring.scaling.max_power_w')
        if max_w < idle_w:
            raise ValueError('ring.scaling.max_power_w deve essere >= idle_power_w')
        if self.boot_power_w is not None:
            nonnegative_number(self.boot_power_w, 'ring.scaling.boot_power_w')
        if self.drain_power_w is not None:
            nonnegative_number(self.drain_power_w, 'ring.scaling.drain_power_w')
        nonnegative_number(self.allocation_cost_per_s, 'ring.scaling.allocation_cost_per_s')
        nonnegative_number(self.cooldown_s, 'ring.scaling.cooldown_s')

    @property
    def effective_boot_power_w(self) -> float:
        """Potenza assorbita durante l'acquisizione e il boot del nodo (W)."""
        return self.idle_power_w if self.boot_power_w is None else self.boot_power_w

    @property
    def effective_drain_power_w(self) -> float:
        """Potenza di base assorbita durante il drain del nodo (W)."""
        return self.idle_power_w if self.drain_power_w is None else self.drain_power_w

    @classmethod
    def parse(cls, data: object, *, prefix: str = 'ring.scaling') -> 'RingScalingConfig':
        """Parse della configurazione di scaling di un singolo ring."""
        if data is None:
            return cls()
        allowed = {
            'min_nodes', 'max_nodes', 'boot_time_s', 'shutdown_time_s',
            'provision_failure_prob', 'idle_power_w', 'max_power_w',
            'boot_power_w', 'drain_power_w', 'allocation_cost_per_s', 'cooldown_s',
        }
        if not isinstance(data, Mapping) or data.keys() - allowed:
            raise ValueError(f'{prefix} ammette solo: {", ".join(sorted(allowed))}')
        return cls(
            min_nodes=nonnegative_integer(data.get('min_nodes', 0), f'{prefix}.min_nodes'),
            max_nodes=(
                positive_integer(data['max_nodes'], f'{prefix}.max_nodes')
                if 'max_nodes' in data and data['max_nodes'] is not None else None
            ),
            boot_time_s=nonnegative_number(data.get('boot_time_s', 5.0), f'{prefix}.boot_time_s'),
            shutdown_time_s=nonnegative_number(data.get('shutdown_time_s', 0.0), f'{prefix}.shutdown_time_s'),
            provision_failure_prob=nonnegative_number(
                data.get('provision_failure_prob', 0.0), f'{prefix}.provision_failure_prob',
            ),
            idle_power_w=nonnegative_number(data.get('idle_power_w', 50.0), f'{prefix}.idle_power_w'),
            max_power_w=nonnegative_number(data.get('max_power_w', 150.0), f'{prefix}.max_power_w'),
            boot_power_w=(
                nonnegative_number(data['boot_power_w'], f'{prefix}.boot_power_w')
                if 'boot_power_w' in data and data['boot_power_w'] is not None else None
            ),
            drain_power_w=(
                nonnegative_number(data['drain_power_w'], f'{prefix}.drain_power_w')
                if 'drain_power_w' in data and data['drain_power_w'] is not None else None
            ),
            allocation_cost_per_s=nonnegative_number(
                data.get('allocation_cost_per_s', 0.0), f'{prefix}.allocation_cost_per_s',
            ),
            cooldown_s=nonnegative_number(data.get('cooldown_s', 0.0), f'{prefix}.cooldown_s'),
        )
