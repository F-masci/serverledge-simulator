"""Core riusabile dell'autoscaler regionale (indipendente dal simulatore)."""

from typing import Any

from .config import PPOConfig, RewardConfig
from .contracts import (
    CommandOutcome, FEATURES_PER_RING, RingObservation, ScalingActionKind,
    ScalingBackend, ScalingCommand, ScalingPolicy, ScalingResult, ZoneObservation,
)
from .core import (
    ActionCodec, ObservationEncoder, RewardBreakdown, RewardCalculator, ZoneController,
)

_LAZY_EXPORTS = {
    'ActionMaskingPPOTorchRLModule': ('.rllib.module', 'ActionMaskingPPOTorchRLModule'),
    'RLlibPPOPolicy': ('.rllib.policy', 'RLlibPPOPolicy'),
    'load_rllib_module': ('.rllib.policy', 'load_rllib_module'),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        from importlib import import_module
        module_name, attr_name = _LAZY_EXPORTS[name]
        return getattr(import_module(module_name, __name__), attr_name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = [
    'ActionCodec',
    'ActionMaskingPPOTorchRLModule',
    'CommandOutcome',
    'FEATURES_PER_RING',
    'ObservationEncoder',
    'PPOConfig',
    'RLlibPPOPolicy',
    'RewardBreakdown',
    'RewardCalculator',
    'RewardConfig',
    'RingObservation',
    'ScalingActionKind',
    'ScalingBackend',
    'ScalingCommand',
    'ScalingPolicy',
    'ScalingResult',
    'ZoneController',
    'ZoneObservation',
    'load_rllib_module',
]
