"""Distribuzioni degli interarrivi e delle durate di servizio, separate dalle funzioni."""

from abc import ABC, abstractmethod

from sim.utils.validation import positive_number
from .streams import RandomStreams, StreamKind


#########################
### Arrival processes ###
#########################

class ArrivalProcess(ABC):
    """Contratto comune per i processi di generazione degli interarrivi."""

    @abstractmethod
    def next_iat(self) -> float:
        """Restituisce l'intervallo in secondi prima del prossimo arrivo."""


class PoissonArrivalProcess(ArrivalProcess):
    """Processo di arrivi Poisson omogeneo con tasso espresso in richieste al secondo."""

    def __init__(self, rate: float, streams: RandomStreams, stream_id: str) -> None:
        # Tasso medio di arrivo e interarrivo medio (parametro di scala esponenziale).
        self._rate = positive_number(rate, 'rate')
        self._scale = positive_number(1 / self._rate, 'interarrivo medio')

        # Stream PRNG dedicato agli arrivi della funzione.
        self._rng = streams.stream(StreamKind.ARRIVAL, stream_id)

    ### Parameters ###

    @property
    def rate(self) -> float:
        return self._rate

    ### Sampling ###

    def next_iat(self) -> float:
        """Campiona un interarrivo esponenziale di media 1 / rate."""
        return float(self._rng.exponential(self._scale))


class DeterministicArrivalProcess(ArrivalProcess):
    """Processo di arrivi deterministico con intervallo costante pari a 1 / rate."""

    def __init__(self, rate: float) -> None:
        # Tasso costante e intervallo fisso tra arrivi consecutivi.
        self._rate = positive_number(rate, 'rate')
        self._interval = positive_number(1 / self._rate, 'interarrivo medio')

    ### Parameters ###

    @property
    def rate(self) -> float:
        return self._rate

    ### Sampling ###

    def next_iat(self) -> float:
        """Restituisce l'interarrivo fisso in secondi."""
        return self._interval


##################################
### Service time distributions ###
##################################

class ServiceTime(ABC):
    """Contratto comune per le distribuzioni delle durate di servizio."""

    @abstractmethod
    def sample(self) -> float:
        """Restituisce la durata di servizio in secondi, esclusa inizializzazione e attesa."""


class GammaServiceTime(ServiceTime):
    """
    Distribuzione Gamma delle durate di servizio parametrizzata da media e SCV (varianza / media**2).

    I parametri descrivono un profilo prestazionale: non derivano automaticamente
    da CPU demand, architettura o numero di invocazioni concorrenti.
    """

    def __init__(
        self, mean: float, scv: float, streams: RandomStreams, stream_id: str,
    ) -> None:
        # Parametri statistici e traduzione nei parametri (shape, scale) della Gamma.
        self._mean = positive_number(mean, 'mean')
        self._scv = positive_number(scv, 'scv')
        self._shape = positive_number(1 / self._scv, 'shape')
        self._scale = positive_number(self._mean * self._scv, 'scale')

        # Stream PRNG dedicato alle durate di servizio della funzione.
        self._rng = streams.stream(StreamKind.SERVICE, stream_id)

    ### Parameters ###

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def scv(self) -> float:
        return self._scv

    ### Sampling ###

    def sample(self) -> float:
        """Campiona una durata di servizio secondo Gamma(shape, scale)."""
        return float(self._rng.gamma(shape=self._shape, scale=self._scale))


class DeterministicServiceTime(ServiceTime):
    """Durata di servizio deterministica e costante."""

    def __init__(self, mean: float) -> None:
        # Durata fissa di servizio in secondi.
        self._duration = positive_number(mean, 'mean')

    ### Parameters ###

    @property
    def mean(self) -> float:
        return self._duration

    ### Sampling ###

    def sample(self) -> float:
        """Restituisce la durata di servizio costante in secondi."""
        return self._duration
