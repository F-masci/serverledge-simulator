"""Contratto pubblico delle notifiche di osservabilità."""

from abc import ABC, abstractmethod
from enum import StrEnum


class NotificationType(StrEnum):
    """Catalogo delle notifiche attualmente pubblicabili.

    CONTAINER_STARTED segnala che la creazione è riuscita e il container è pronto.
    CONTAINER_STOPPED segnala la rimozione di un container già avviato.
    SIMULATION_ERROR segnala il fallimento di un evento a calendario.
    FINAL_STATISTICS trasporta le metriche al termine di una replica.
    """

    CONTAINER_STARTED = 'container.started'
    CONTAINER_STOPPED = 'container.stopped'
    CONTAINER_STATE_CHANGED = 'container.state_changed'
    NODE_REGISTERED = 'node.registered'
    SIMULATION_ERROR = 'simulation.error'
    REQUEST_ARRIVED = 'request.arrived'
    REQUEST_QUEUED = 'request.queued'
    REQUEST_FORWARDED = 'request.forwarded'
    REQUEST_DELIVERED = 'request.delivered'
    REQUEST_ACCEPTED = 'request.accepted'
    REQUEST_REJECTED = 'request.rejected'
    REQUEST_COMPLETED = 'request.completed'
    FINAL_STATISTICS = 'simulation.final_statistics'


class Notification(ABC):
    """Notifica immutabile esportabile come messaggio JSON."""

    @property
    @abstractmethod
    def type(self) -> NotificationType:
        """Identificatore stabile del tipo di notifica."""

    @property
    @abstractmethod
    def level(self) -> int:
        """Livello standard di logging dichiarato dalla classe concreta."""

    @abstractmethod
    def __str__(self) -> str:
        """Restituisce il messaggio leggibile usato dal logger."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Restituisce il payload serializzabile in JSON per i subscriber esterni."""
