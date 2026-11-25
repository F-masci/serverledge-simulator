"""Validatori autonomi per i parametri di configurazione dello scaler."""

import math
from numbers import Integral, Real


def nonnegative_number(value: float, name: str) -> float:
    """Verifica che il valore sia un numero reale finito e non negativo (>= 0)."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} deve essere un numero finito e non negativo')
    return float(value)


def positive_number(value: float, name: str) -> float:
    """Verifica che il valore sia un numero reale finito e strettamente positivo (> 0)."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} deve essere finito e positivo')
    return float(value)


def positive_integer(value: int, name: str) -> int:
    """Verifica che il valore sia un intero strettamente positivo (> 0)."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f'{name} deve essere un intero positivo')
    return int(value)
