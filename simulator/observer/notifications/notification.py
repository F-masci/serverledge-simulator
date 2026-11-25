"""Contratto pubblico delle notifiche di osservabilità."""

import json
from abc import ABC, abstractmethod
from enum import StrEnum


class NotificationType(StrEnum):
    """Catalogo delle notifiche pubblicabili dall'observer."""
    CONTAINER_STARTED = 'container.started'              # Creazione riuscita: il container è pronto.
    CONTAINER_STOPPED = 'container.stopped'              # Rimozione di un container già avviato.
    CONTAINER_STATE_CHANGED = 'container.state_changed'  # Transizione di stato del container con snapshot risorse.
    NODE_REGISTERED = 'node.registered'                            # Registrazione di un nodo nella replica.
    NODE_REMOVED = 'node.removed'                                  # Rimozione di un nodo dalla topologia (scale-in).
    SIMULATION_ERROR = 'simulation.error'                # Fallimento di un evento o della simulazione.
    REQUEST_ARRIVED = 'request.arrived'                  # Arrivo di una nuova invocazione nella regione di ingresso.
    REQUEST_QUEUED = 'request.queued'                    # Accodamento della richiesta nella FIFO del nodo.
    REQUEST_DEQUEUED = 'request.dequeued'                # Estrazione della richiesta dalla FIFO del nodo.
    REQUEST_FORWARDED = 'request.forwarded'              # Partenza della richiesta verso una regione remota.
    REQUEST_DELIVERED = 'request.delivered'              # Consegna della richiesta inoltrata nella regione remota.
    REQUEST_ACCEPTED = 'request.accepted'                          # Ammissione della richiesta su container o regione.
    REQUEST_REJECTION_RETURNING = 'request.rejection_returning'    # Rifiuto remoto in transito verso il client.
    REQUEST_REJECTED = 'request.rejected'                          # Rifiuto definitivo della richiesta.
    REQUEST_COMPLETED = 'request.completed'              # Consegna della risposta e completamento della richiesta.
    SCALER_DECISION = 'scaler.decision'                  # Decisione periodica dello scaler regionale.
    FINAL_STATISTICS = 'simulation.final_statistics'     # Fotografia delle metriche al termine della replica.


class Notification(ABC):
    """Contratto base delle notifiche immutabili esportabili come messaggio JSON."""

    ### Identification ###

    @property
    @abstractmethod
    def type(self) -> NotificationType:
        """Identificatore stabile del tipo di notifica."""

    @property
    @abstractmethod
    def level(self) -> int:
        """Livello standard di logging dichiarato dalla classe concreta."""

    #####################
    ### Serialization ###
    #####################

    @abstractmethod
    def __str__(self) -> str:
        """Restituisce il messaggio leggibile usato dal logger."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Restituisce il payload serializzabile in JSON per i subscriber esterni."""

    def to_json(self) -> str:
        """Serializza la notifica in JSON per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)
