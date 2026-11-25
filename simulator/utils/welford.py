"""Calcolo online di media, varianza e medie temporali di segnali a tratti costanti."""

from copy import copy
from dataclasses import dataclass


@dataclass
class Welford:
    """Accumulatore online di media e varianza pesata con algoritmo di Welford."""

    weight: float = 0.0
    _mean: float = 0.0
    _m2: float = 0.0

    ### Statistics ###

    @property
    def mean(self) -> float | None:
        """Restituisce la media pesata osservata, o None se nessun peso è stato accumulato."""
        return self._mean if self.weight else None

    @property
    def variance(self) -> float | None:
        """Restituisce la varianza della popolazione osservata, con i pesi indicati."""
        return self._m2 / self.weight if self.weight else None

    #################################
    ### Internal state management ###
    #################################

    def add(self, value: float, weight: float = 1.0) -> None:
        """Aggiorna media e somma degli scarti quadratici con un nuovo campione pesato."""
        if weight < 0:
            raise ValueError('Il peso non può essere negativo')
        if weight == 0:
            return

        # Aggiornamento numericamente stabile secondo Welford pesato.
        self.weight += weight
        delta = value - self._mean
        self._mean += delta * (weight / self.weight)
        self._m2 += weight * delta * (value - self._mean)


class TimedWelford:
    """Accumulatore temporale che pesa ciascun valore per la sua durata di permanenza."""

    def __init__(self, now: float, value: float) -> None:
        # Istante e valore dell'ultima osservazione registrata.
        self._now = now
        self._value = value
        self._moments = Welford()

    #################################
    ### Internal state management ###
    #################################

    def observe(self, now: float, value: float) -> None:
        """Chiude l'intervallo precedente fino a now e registra il nuovo valore corrente."""
        self._moments.add(self._value, now - self._now)
        self._now = now
        self._value = value

    def snapshot(self, now: float) -> Welford:
        """Restituisce i momenti inclusivi dell'intervallo corrente senza mutare lo stato."""
        moments = copy(self._moments)
        moments.add(self._value, now - self._now)
        return moments
