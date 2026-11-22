"""Validatori condivisi per parametri numerici, interi, stringhe e domini."""

import math
from collections.abc import Sequence
from numbers import Integral, Real


##########################
### Numeric validators ###
##########################

def finite_number(value: float, name: str) -> float:
    """Verifica che il valore sia un numero reale finito (escludendo i booleani)."""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} deve essere un numero finito')
    return float(value)


def nonnegative_number(value: float, name: str) -> float:
    """Verifica che il valore sia un numero reale finito e non negativo (>= 0)."""
    number = finite_number(value, name)
    if number < 0:
        raise ValueError(f'{name} deve essere un numero finito e non negativo')
    return number


def positive_number(value: float, name: str) -> float:
    """Verifica che il valore sia un numero reale finito e strettamente positivo (> 0)."""
    number = finite_number(value, name)
    if number <= 0:
        raise ValueError(f'{name} deve essere finito e positivo')
    return number


##########################
### Integer validators ###
##########################

def integer(value: int, name: str) -> int:
    """Verifica che il valore sia un intero (escludendo i booleani)."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} deve essere un intero')
    return int(value)


def nonnegative_integer(value: int, name: str) -> int:
    """Verifica che il valore sia un intero non negativo (>= 0)."""
    number = integer(value, name)
    if number < 0:
        raise ValueError(f'{name} deve essere un intero non negativo')
    return number


def positive_integer(value: int, name: str) -> int:
    """Verifica che il valore sia un intero strettamente positivo (> 0)."""
    number = nonnegative_integer(value, name)
    if number == 0:
        raise ValueError(f'{name} deve essere positivo')
    return number


####################################
### String and domain validators ###
####################################

def nonempty_string(value: str, name: str) -> str:
    """Verifica che il valore sia una stringa non vuota (esclusi soli spazi)."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} deve essere una stringa non vuota')
    return value


def one_of[T](value: T, allowed: Sequence[T], name: str) -> T:
    """Verifica che il valore appartenga alla sequenza dei valori ammessi."""
    if value not in allowed:
        choices = ', '.join(map(str, allowed))
        raise ValueError(f'{name} deve essere uno tra {choices}')
    return value


###############
### Aliases ###
###############

# Alias semantici per istanti temporali e ritardi del motore a eventi discreti.
finite_time = finite_number
finite_delay = nonnegative_number
