"""Richiesta di invocazione e risultato dell'ammissione sul nodo."""

import json
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING
from uuid import uuid4

from .function import Function

if TYPE_CHECKING:
    from .container import DockerContainer
    from .node import Node
    from .region import Region
    from .infrastructure import Infrastructure
    from .ring import ArchitectureRing


class RequestStatus(Enum):
    """Stati di una richiesta durante ammissione ed esecuzione."""
    CREATED = 'created'        # Pronta per l'ammissione, anche dopo l'arrivo nella regione remota.
    IN_TRANSIT = 'in_transit'  # In viaggio verso la regione di destinazione.
    QUEUED = 'queued'          # In attesa di risorse nella FIFO del nodo.
    ACCEPTED = 'accepted'      # Ammessa, in attesa del cold start o della fine del servizio.
    REJECTED = 'rejected'      # Rifiutata definitivamente durante il routing o l'ammissione.
    RESPONDING = 'responding'  # Servizio terminato, risorse rilasciate, risposta in viaggio.
    COMPLETED = 'completed'    # Risposta consegnata al client e completamento registrato.


class Request:
    """Rappresenta una singola invocazione di una funzione."""

    def __init__(
        self, function: Function, *, service_duration: float | None = None, arrival_time: float = 0.0,
        can_offload: bool = False,
    ) -> None:
        # Identificativo univoco della richiesta
        self._id = uuid4().hex
        # Stato della richiesta, inizialmente in attesa di ammissione
        self._status = RequestStatus.CREATED

        # Parametri della funzione richiesta
        self._function = function
        self._arrival_time = arrival_time
        self._service_duration = service_duration

        # Risorse che gestiscono la richiesta
        self._region: Region | None = None
        self._node: Node | None = None
        self._container: DockerContainer | None = None
        self._warm_execution: bool | None = None
        self._rejection_reason: str | None = None

        # Stato di offloading e routing
        self._allow_offloading = can_offload
        self._offloaded_from: Node | None = None
        self._entry_region: Region | None = None
        self._region_arrival_time = arrival_time
        self._return_delay = 0.0

        ### Callback ###
        self.on_queued: Callable[[Request], None] | None = None
        self.on_accepted: Callable[[Request], None] | None = None
        self.on_rejected: Callable[[Request], None] | None = None
        self.on_completed: Callable[[Request], None] | None = None
        self.on_state_changed: Callable[[Request, RequestStatus, RequestStatus], None] | None = None

    ### Identification ###

    @property
    def id(self) -> str:
        """Identificativo della singola invocazione."""
        return self._id

    ### Infrastructure ###

    @property
    def container(self) -> 'DockerContainer | None':
        return self._container

    @property
    def node(self) -> 'Node | None':
        node = self._node
        container = self.container
        if node is None and container is not None:
            node = container.node
        return node

    @property
    def ring(self) -> ArchitectureRing:
        node = self.node
        assert node is not None, 'Richiesta non associata a un nodo'
        return node.ring

    @property
    def infrastructure(self) -> Infrastructure:
        return self.ring.infrastructure

    @property
    def region(self) -> 'Region | None':
        region = self._region
        node = self.node
        if region is None and node is not None:
            region = node.region
        return region

    ### Request State ###

    @property
    def status(self) -> RequestStatus:
        return self._status

    @property
    def is_created(self) -> bool:
        """Indica se la richiesta è stata creata e non ancora ammessa."""
        return self._status is RequestStatus.CREATED

    @property
    def is_accepted(self) -> bool:
        """Indica se la richiesta è stata ammessa su un container o una regione."""
        return self._status is RequestStatus.ACCEPTED

    @property
    def is_rejected(self) -> bool:
        """Indica se la richiesta è stata rifiutata definitivamente."""
        return self._status is RequestStatus.REJECTED

    @property
    def is_completed(self) -> bool:
        """Indica se la richiesta è stata completata con risposta consegnata."""
        return self._status is RequestStatus.COMPLETED

    @property
    def is_responding(self) -> bool:
        """Indica se la richiesta ha completato il servizio e sta tornando la risposta."""
        return self._status is RequestStatus.RESPONDING

    @property
    def is_queued(self) -> bool:
        """Indica se la richiesta è in attesa di ammissione nella FIFO del nodo."""
        return self._status is RequestStatus.QUEUED

    @property
    def is_in_transit(self) -> bool:
        """Indica se la richiesta è in viaggio verso la regione di destinazione."""
        return self._status is RequestStatus.IN_TRANSIT

    @property
    def is_container_accepted(self) -> bool:
        """Indica se la richiesta è stata ammessa su un container."""
        return self._status is RequestStatus.ACCEPTED and self._container is not None

    @property
    def is_region_accepted(self) -> bool:
        """Indica se la richiesta è stata ammessa su una regione senza container."""
        return self._status is RequestStatus.ACCEPTED and self._container is None

    ### Execution State ###

    @property
    def function(self) -> Function:
        return self._function

    @property
    def service_duration(self) -> float | None:
        """Durata di servizio richiesta, in secondi simulati."""
        return self._service_duration

    @property
    def arrival_time(self) -> float:
        return self._arrival_time

    @property
    def warm(self) -> bool | None:
        return self._warm_execution

    @property
    def rejection_reason(self) -> str | None:
        return self._rejection_reason

    @property
    def can_offload(self) -> bool:
        return self._allow_offloading and self._offloaded_from is None

    @property
    def region_arrival_time(self) -> float:
        return self._region_arrival_time

    @property
    def return_delay(self) -> float:
        return self._return_delay

    #################################
    ### Internal state management ###
    #################################

    ## Infra management ##

    def enter_region(self, region: 'Region', now: float) -> None:
        if self._status is not RequestStatus.IN_TRANSIT:
            self.ensure_created()
        self._region = region
        if self._entry_region is None:
            self._entry_region = region
        self._region_arrival_time = now
        if self._status is RequestStatus.IN_TRANSIT:
            self._handle_change_status(RequestStatus.CREATED, None)

    def start_transfer(self, region: 'Region', return_delay: float) -> None:
        self.ensure_created()
        self._region = region
        self._return_delay = return_delay
        self._handle_change_status(RequestStatus.IN_TRANSIT, None)

    def mark_offloaded(self, source: 'Node') -> None:
        self.ensure_created()
        if not self.can_offload:
            raise ValueError('Offloading non consentito per questa richiesta')
        self._offloaded_from = source

    ## Request lifecycle management ##

    def enqueue(self, node: Node) -> None:
        """Assegna la richiesta alla FIFO del nodo."""
        self.ensure_created()

        # Conserva il nodo che potrà ammettere la richiesta dalla coda.
        self._node = node
        self._handle_change_status(RequestStatus.QUEUED, self.on_queued)

    def accept_on_container(self, container: DockerContainer) -> None:
        """Ammette una nuova richiesta su un container specifico."""
        self.ensure_admissible()

        # Conserva il nodo e il container che gestiranno la richiesta.
        self._container = container
        self._warm_execution = container.is_warm
        self._node = container.node
        self._handle_change_status(RequestStatus.ACCEPTED, self.on_accepted)

    def accept_on_region(self, region: Region) -> None:
        """Ammette una nuova richiesta su un'infrastruttura di regione."""
        self.ensure_admissible()

        # Conserva la regione che gestirà la richiesta.
        self.ensure_created()
        self._region = region
        self._service_duration = region.service_time_s
        self._handle_change_status(RequestStatus.ACCEPTED, self.on_accepted)

    def reject(self, reason: str, *, node: 'Node | None' = None) -> None:
        """Registra il rifiuto definitivo e l'eventuale nodo destinatario."""
        self.ensure_created()

        # Conserva il nodo che ha rifiutato la richiesta, se noto.
        self._node = node
        self._rejection_reason = reason
        self._handle_change_status(RequestStatus.REJECTED, self.on_rejected)

    def complete(self, now: float) -> None:
        """Conclude un'invocazione con risposta immediata."""
        self._release_execution(now)
        self._handle_change_status(RequestStatus.COMPLETED, self.on_completed)

    def begin_response(self, now: float) -> None:
        """Inizia il ritorno della risposta, rilasciando le risorse."""
        self._release_execution(now)
        self._handle_change_status(RequestStatus.RESPONDING, None)

    def finish_response(self) -> None:
        """Conclude un'invocazione con risposta in transito."""
        self.ensure_responding()
        self._handle_change_status(RequestStatus.COMPLETED, self.on_completed)

    def _release_execution(self, now: float) -> None:
        """Rilascia le risorse di esecuzione della richiesta."""
        self.ensure_accepted()

        # Rilascia il container e notifica il nodo che la richiesta è completata.
        #
        # Il nodo e il container possono essere None se la
        # richiesta è stata ammessa direttamente su una
        # regione senza container.
        if self.is_container_accepted:
            container = self.container
            node = self.node
            assert container is not None and node is not None
            node.complete(container, now)

    ## Callback handling ##

    def _handle_change_status(self, status: RequestStatus, callback: Callable[[Request], None] | None) -> None:
        """Aggiorna lo stato prima di invocare i callback."""
        previous_status = self._status
        self._status = status

        # Notifica prima la transizione specifica, poi il cambio di stato generico.
        if callback is not None:
            callback(self)
        if self.on_state_changed is not None:
            self.on_state_changed(self, previous_status, status)

    ######################
    ### Error handling ###
    ######################

    def _ensure_status(self, expected: RequestStatus, message: str) -> None:
        if self._status is not expected:
            raise ValueError(f'{message}: stato attuale={self._status.value}, atteso={expected.value}')

    def _ensure_statuses(self, expected: tuple[RequestStatus, ...], message: str) -> None:
        if self._status not in expected:
            expected_values = ', '.join(s.value for s in expected)
            raise ValueError(f'{message}: stato attuale={self._status.value}, attesi={expected_values}')

    def ensure_created(self) -> None:
        """Controlla che la richiesta sia ancora in attesa di ammissione."""
        self._ensure_status(RequestStatus.CREATED, 'La richiesta non è più in attesa di ammissione')

    def ensure_admissible(self) -> None:
        """Controlla che la richiesta sia ancora in attesa di ammissione o in viaggio verso la regione."""
        self._ensure_statuses((RequestStatus.CREATED, RequestStatus.QUEUED, RequestStatus.IN_TRANSIT), 'La richiesta non è più ammissibile')

    def ensure_accepted(self) -> None:
        """Controlla che la richiesta sia stata ammessa e sia in esecuzione."""
        self._ensure_status(RequestStatus.ACCEPTED, 'La richiesta non è più in esecuzione')

    def ensure_responding(self) -> None:
        """Controlla che la richiesta sia in fase di ritorno della risposta."""
        self._ensure_status(RequestStatus.RESPONDING, 'La richiesta non ha una risposta in transito')

    def ensure_completed(self) -> None:
        """Controlla che la richiesta sia stata completata."""
        self._ensure_status(RequestStatus.COMPLETED, 'La richiesta non è ancora completata')

    #####################
    ### Serialization ###
    #####################


    def __str__(self) -> str:
        """Restituisce una descrizione leggibile della richiesta."""
        fields = self.to_dict()
        details = ', '.join(f'{name}={value}' for name, value in fields.items() if value is not None)
        return f'Request({details})'

    def to_dict(self) -> dict:
        """Restituisce lo stato della richiesta con i riferimenti espressi come ID."""
        node = self._node
        container = self._container
        return {
            'id': self._id,
            'function': self._function.name,
            'status': self._status.value,
            'arrival_time_s': self._arrival_time,
            'service_duration_s': self._service_duration,
            'node_id': node.id.key if node is not None else None,
            'container_id': container.id if container is not None else None,
            'warm': self._warm_execution,
            'rejection_reason': self._rejection_reason,
            'entry_region': self._entry_region.name if self._entry_region is not None else None,
            'region': self._region.name if self._region is not None else None,
            'ring_id': self.ring.id if node is not None else None,
            'offloaded_from_node_id': self._offloaded_from.id.key if self._offloaded_from is not None else None,
        }

    def to_json(self) -> str:
        """Serializza la richiesta per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<Request {self._id} function={self._function.name} status={self._status.value}>'