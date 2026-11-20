"""Validatori numerici condivisi dal motore e dai generatori casuali."""

import math
from collections.abc import Sequence
from numbers import Integral, Real


def finite_time(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} deve essere un numero finito')
    return float(value)

def finite_delay(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} deve essere un numero finito e non negativo')
    return float(value)

def positive_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} deve essere finito e positivo')
    return float(value)


def integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} deve essere un intero')
    return int(value)


def nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f'{name} deve essere un intero non negativo')
    return int(value)

def nonnegative_number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} deve essere un numero finito e non negativo')
    return float(value)

def nonempty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} deve essere una stringa non vuota')
    return value


def positive_integer(value: int, name: str) -> int:
    value = nonnegative_integer(value, name)
    if value == 0:
        raise ValueError(f'{name} deve essere positivo')
    return value


def one_of[T](value: T, allowed: Sequence[T], name: str) -> T:
    """Verifica che il valore appartenga alla sequenza dei valori ammessi."""
    if value not in allowed:
        choices = ', '.join(map(str, allowed))
        raise ValueError(f'{name} deve essere uno tra {choices}')
    return value
