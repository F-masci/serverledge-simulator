"""Modelli di topologia infrastrutturale: nodi, anelli, infrastruttura e regioni."""

from typing import Any

__all__ = [
    'ArchitectureRing',
    'Infrastructure',
    'Node',
    'NodeID',
    'Region',
    'RegionMode',
    'Resources',
]

_EXPORTS = {
    'ArchitectureRing': ('.ring', 'ArchitectureRing'),
    'Infrastructure': ('.infrastructure', 'Infrastructure'),
    'Node': ('.node', 'Node'),
    'NodeID': ('.node', 'NodeID'),
    'Region': ('.region', 'Region'),
    'RegionMode': ('.region', 'RegionMode'),
    'Resources': ('.node', 'Resources'),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        from importlib import import_module

        module_name, attr_name = _EXPORTS[name]
        return getattr(import_module(module_name, __name__), attr_name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
