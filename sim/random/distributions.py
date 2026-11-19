"""Distribuzioni dei tempi, separate dalle configurazioni delle funzioni."""

from abc import ABC, abstractmethod

from sim.utils.validation import positive_number

from .streams import RandomStreams, StreamKind


class ArrivalProcess(ABC):
    @abstractmethod
    def next_iat(self) -> float:
        """Intervallo in secondi prima del prossimo arrivo."""


class PoissonArrivalProcess(ArrivalProcess):
    """Arrivi Poisson omogenei con tasso espresso in richieste al secondo."""

    def __init__(self, rate: float, streams: RandomStreams, stream_id: str) -> None:
        self._rate = positive_number(rate, 'rate')
        self._scale = positive_number(1 / self._rate, 'interarrivo medio')
        self._rng = streams.stream(StreamKind.ARRIVAL, stream_id)

    @property
    def rate(self) -> float:
        return self._rate

    def next_iat(self) -> float:
        return float(self._rng.exponential(self._scale))


class DeterministicArrivalProcess(ArrivalProcess):
    def __init__(self, rate: float) -> None:
        self._interval = positive_number(1 / positive_number(rate, 'rate'), 'interarrivo medio')

    def next_iat(self) -> float:
        return self._interval


class ServiceTime(ABC):
    @abstractmethod
    def sample(self) -> float:
        """Durata di servizio in secondi, esclusa inizializzazione e attesa."""


class GammaServiceTime(ServiceTime):
    """Servizio Gamma parametrizzato da media e SCV = varianza / media**2.

    I parametri descrivono un profilo prestazionale: non derivano automaticamente
    da CPU demand, architettura o numero di invocazioni concorrenti.
    """

    def __init__(
        self, mean: float, scv: float, streams: RandomStreams, stream_id: str,
    ) -> None:
        self._mean = positive_number(mean, 'mean')
        self._scv = positive_number(scv, 'scv')
        self._shape = positive_number(1 / self._scv, 'shape')
        self._scale = positive_number(self._mean * self._scv, 'scale')
        self._rng = streams.stream(StreamKind.SERVICE, stream_id)

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def scv(self) -> float:
        return self._scv

    def sample(self) -> float:
        return float(self._rng.gamma(shape=self._shape, scale=self._scale))


class DeterministicServiceTime(ServiceTime):
    def __init__(self, mean: float) -> None:
        self._duration = positive_number(mean, 'mean')

    def sample(self) -> float:
        return self._duration
