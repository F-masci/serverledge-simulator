"""Momenti online e media temporale di segnali a tratti costanti."""

from copy import copy
from dataclasses import dataclass


@dataclass
class Welford:
    weight: float = 0.0
    _mean: float = 0.0
    _m2: float = 0.0

    def add(self, value: float, weight: float = 1.0) -> None:
        if weight < 0:
            raise ValueError('Il peso non può essere negativo')
        if weight == 0:
            return
        self.weight += weight
        delta = value - self._mean
        self._mean += delta * (weight / self.weight)
        self._m2 += weight * delta * (value - self._mean)

    @property
    def mean(self) -> float | None:
        return self._mean if self.weight else None

    @property
    def variance(self) -> float | None:
        """Varianza della popolazione osservata, con i pesi indicati."""
        return self._m2 / self.weight if self.weight else None


class TimedWelford:
    """Pesa il valore corrente per la durata fino al prossimo cambiamento."""

    def __init__(self, now: float, value: float) -> None:
        self._now = now
        self._value = value
        self._moments = Welford()

    def observe(self, now: float, value: float) -> None:
        self._moments.add(self._value, now - self._now)
        self._now = now
        self._value = value

    def snapshot(self, now: float) -> Welford:
        """Include l'intervallo corrente senza modificare l'accumulatore."""
        moments = copy(self._moments)
        moments.add(self._value, now - self._now)
        return moments
