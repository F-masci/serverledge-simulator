"""Validatori numerici condivisi dal motore e dai generatori casuali."""

import math
from numbers import Integral, Real


def finite_time(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} deve essere un numero finito')
    return float(value)


def positive_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} deve essere finito e positivo')
    return float(value)


def nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} deve essere un intero non negativo')
    return int(value)
