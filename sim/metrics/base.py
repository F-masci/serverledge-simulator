"""Contratto comune dei collector di metriche alimentati dall'observer."""

from abc import ABC, abstractmethod
import json
import math

from sim.observer.notifications import Notification, NotificationType
from sim.utils.welford import TimedWelford

# Chiave univoca di un nodo nelle metriche: (regione, ring_id, node_id).
type NodeKey = tuple[str, str, str]

# Chiave univoca di un anello nelle metriche: (regione, ring_id).
type RingKey = tuple[str, str]


##########################
### Statistical helper ###
##########################

def timed_snapshot(accumulator: TimedWelford, now: float) -> dict:
    """Restituisce i momenti temporali (peso, media, varianza) all'istante now."""
    moments = accumulator.snapshot(now)
    return {
        'observed_time_s': moments.weight,
        'mean': moments.mean,
        'variance': moments.variance,
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    """Restituisce numerator / denominator, oppure None se il denominatore è nullo."""
    return numerator / denominator if denominator > 0 else None


def _percentile(sorted_values: list[float], p: float) -> float:
    """Calcola il percentile p in [0, 100] tramite interpolazione lineare."""
    rank = (p / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])


def latency_percentiles(samples: list[float]) -> dict[str, float | None]:
    """Calcola min, p50, p90, p95, p99 e max sui campioni osservati."""
    if not samples:
        return {'min': None, 'p50': None, 'p90': None, 'p95': None, 'p99': None, 'max': None}
    ordered = sorted(samples)
    return {
        'min': ordered[0],
        'p50': _percentile(ordered, 50),
        'p90': _percentile(ordered, 90),
        'p95': _percentile(ordered, 95),
        'p99': _percentile(ordered, 99),
        'max': ordered[-1],
    }


########################
### Collector contract ###
########################

class MetricCollector(ABC):
    """
    Interfaccia comune per i collector specializzati di metriche.

    Ogni collector dichiara i tipi di notifica di proprio interesse e aggiorna
    il proprio stato esclusivamente tramite handle_notification(notification),
    esponendo metodi di serializzazione e formattazione testuale indipendenti.
    """

    ### Identification ###

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificatore univoco del gruppo di metriche gestito dal collector."""

    @property
    @abstractmethod
    def notification_types(self) -> tuple[NotificationType, ...]:
        """Tipi di notifica instradati a questo collector da SimulationMetrics."""

    #############################
    ### Notification handling ###
    #############################

    @abstractmethod
    def handle_notification(self, notification: Notification) -> None:
        """Elabora una notifica estraendo dai riferimenti di dominio i dati di interesse."""

    #####################
    ### Serialization ###
    #####################

    @abstractmethod
    def to_dict(self, now: float = 0.0) -> dict:
        """Restituisce lo snapshot delle metriche del collector in formato serializzabile."""

    def to_json(self, now: float = 0.0) -> str:
        """Serializza in JSON lo snapshot delle metriche del collector."""
        return json.dumps(self.to_dict(now), ensure_ascii=False, allow_nan=False)

    def format_summary(self, now: float = 0.0) -> str:
        """Restituisce una rappresentazione testuale stampabile del singolo collector."""
        return f'[{self.name}]\n{json.dumps(self.to_dict(now), indent=2, ensure_ascii=False, allow_nan=False)}'
