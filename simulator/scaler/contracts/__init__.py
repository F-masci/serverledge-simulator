"""Contratti comuni (DTO e interfacce) dello scaler regionale."""

from .actions import CommandOutcome, ScalingActionKind, ScalingCommand, ScalingResult
from .interfaces import ScalingBackend, ScalingPolicy
from .observations import FEATURES_PER_RING, RingObservation, ZoneObservation

__all__ = [
    'CommandOutcome',
    'FEATURES_PER_RING',
    'RingObservation',
    'ScalingActionKind',
    'ScalingBackend',
    'ScalingCommand',
    'ScalingPolicy',
    'ScalingResult',
    'ZoneObservation',
]
