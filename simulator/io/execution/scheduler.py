"""Configurazione dello scheduler locale condivisa dai nodi.

Esempio TOML nello scenario (tutti i parametri sono opzionali):

    [scheduler]
    policy = "default"               # opzionale (default: "default" | "edgeonly" | "edgecloud" | "cloudonly")
    fallback_local = false           # opzionale (default: false)
    offloading_cache_validity_s = 60 # opzionale (default: 60.0)

    [scheduler.queue]
    capacity = 0                     # opzionale (default: 0, coda disabilitata)
"""

from collections.abc import Mapping
from dataclasses import dataclass

from simulator.model.scheduling.offloading import SchedulingPolicy
from simulator.utils.validation import nonempty_string, nonnegative_integer, nonnegative_number


@dataclass(frozen=True)
class _SchedulerConfig:
    """Parametri della politica di ammissione locale, condivisi dai nodi."""

    queue_capacity: int
    policy: SchedulingPolicy
    fallback_local: bool
    offloading_cache_validity_s: float = 60.0

    @classmethod
    def parse(cls, scheduler: object) -> '_SchedulerConfig':
        """Parse della configurazione dello scheduler."""
        if scheduler is None:
            scheduler = {}

        allowed = {'queue', 'policy', 'fallback_local', 'offloading_cache_validity_s'}
        if not isinstance(scheduler, Mapping) or scheduler.keys() - allowed:
            raise ValueError(f'Campi scheduler ammessi: {", ".join(sorted(allowed))}')
        queue = scheduler.get('queue', {})

        if not isinstance(queue, Mapping) or queue.keys() - {'capacity'}:
            raise ValueError('scheduler.queue accetta solo capacity')
        capacity = nonnegative_integer(queue.get('capacity', 0), 'scheduler.queue.capacity')
        cache_validity = nonnegative_number(
            scheduler.get('offloading_cache_validity_s', 60.0), 'scheduler.offloading_cache_validity_s',
        )
        fallback = scheduler.get('fallback_local', False)
        if not isinstance(fallback, bool):
            raise ValueError('scheduler.fallback_local deve essere un booleano')

        return cls(
            queue_capacity=capacity,
            policy=SchedulingPolicy(nonempty_string(scheduler.get('policy', 'default'), 'scheduler.policy')),
            fallback_local=fallback,
            offloading_cache_validity_s=cache_validity,
        )
