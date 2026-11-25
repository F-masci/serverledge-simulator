"""Configurazioni topologiche: nodi, ring, infrastruttura e regioni."""

from .infrastructure import _InfrastructureConfig
from .node import _NodeConfig
from .region import _RegionConfig
from .ring import _RingConfig

__all__ = ['_InfrastructureConfig', '_NodeConfig', '_RegionConfig', '_RingConfig']
