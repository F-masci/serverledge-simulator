"""Configurazione di ingresso delle invocazioni nello scenario.

Esempio TOML nello scenario (tutti i parametri sono opzionali):

    [invocation]
    entry_region = "edge"   # opzionale (default: primary_region)
    entry_node = "edge-1"   # opzionale (default: None -> instradato dal load balancer)
    can_offload = true      # opzionale (default: false)
"""

from collections.abc import Mapping
from dataclasses import dataclass

from sim.utils.validation import nonempty_string


@dataclass(frozen=True)
class _InvocationConfig:
    """Ingresso e autorizzazione all'offloading per gli arrivi generati."""

    entry_node: str | None = None
    can_offload: bool = False
    entry_region: str | None = None

    @classmethod
    def parse(cls, invocation: object) -> '_InvocationConfig':
        """Parse della configurazione di ingresso delle invocazioni."""
        if invocation is None:
            return cls()
        if not isinstance(invocation, Mapping) or invocation.keys() - {
            'entry_node', 'entry_region', 'can_offload',
        }:
            raise ValueError('invocation accetta solo entry_node, entry_region e can_offload')
        node = nonempty_string(invocation['entry_node'], 'invocation.entry_node') if 'entry_node' in invocation else None
        region = nonempty_string(invocation['entry_region'], 'invocation.entry_region') if 'entry_region' in invocation else None
        can_offload = invocation.get('can_offload', False)
        if not isinstance(can_offload, bool):
            raise ValueError('invocation.can_offload deve essere un booleano')
        return cls(node, can_offload, region)
