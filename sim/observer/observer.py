"""Distribuzione sincrona delle notifiche della replica e del processo."""

from collections.abc import Callable
from itertools import chain
import logging
from typing import TYPE_CHECKING

from .notifications import (
    ContainerStarted, ContainerStateChanged, ContainerStopped, NodeRegistered, Notification,
    RequestAccepted, RequestArrived, RequestCompleted, RequestDelivered, RequestDequeued, RequestForwarded,
    RequestQueued, RequestRejected,
)
from .notifications.node import ContainerSnapshot, ResourceSnapshot
from ..model.container import ContainerState, DockerContainer
from ..model.node import Node
from ..model.request import Request

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext
    from ..model.region import Region


class Observer:
    """
    Dispatcher sincrono delle notifiche di una replica o del processo.

    Costruisce le notifiche mantenendo riferimenti diretti agli oggetti Python
    del modello (regioni, anelli, nodi, container, richieste) e le propaga
    prima ai subscriber locali della replica e poi all'observer padre condiviso.
    """

    def __init__(
        self, ctx: 'SimulationContext | None' = None, *, parent: 'Observer | None' = None,
    ) -> None:
        self._ctx = ctx
        self._parent = parent
        self._subscribers: dict[object, tuple[Callable[[Notification], None], bool]] = {}

    #####################
    ### Subscriptions ###
    #####################

    def subscribe(
        self, handler: Callable[[Notification], None], *, propagate_errors: bool = False,
    ) -> Callable[[], None]:
        """Registra il subscriber e restituisce una funzione per scollegarlo."""
        subscription = object()
        self._subscribers[subscription] = (handler, propagate_errors)

        def unsubscribe() -> None:
            self._subscribers.pop(subscription, None)

        return unsubscribe

    ################
    ### Dispatch ###
    ################

    def notify(self, notification: Notification) -> None:
        """Notifica i subscriber della replica, poi quelli del processo."""
        if not isinstance(notification, Notification):
            raise TypeError('La notifica deve implementare Notification')

        # Itera su una copia perché una callback può iscriversi o scollegarsi durante la notifica.
        for handler, propagate_errors in tuple(self._subscribers.values()):
            try:
                handler(notification)
            except Exception:
                if propagate_errors:
                    raise
                logging.getLogger(__name__).exception('Errore nel subscriber di %s', notification.type)

        if self._parent is not None:
            self._parent.notify(notification)

    def _ensure_context(self) -> 'SimulationContext':
        """Restituisce il contesto associato all'observer o solleva RuntimeError."""
        if self._ctx is None:
            raise RuntimeError('Observer non associato a un contesto di simulazione')
        return self._ctx

    #############################
    ### Request notifications ###
    #############################

    def _request_fields(self, request: Request) -> dict:
        """Raccoglie i campi comuni di contesto e riferimenti di dominio della richiesta."""
        ctx = self._ensure_context()
        node = request.node
        return {
            'run_id': ctx.simulation.run_id,
            'sim_time_s': ctx.engine.now,
            'request': request,
            'region': request.region,
            'ring': node.ring if node is not None else None,
        }

    def notify_arrival(self, request: Request, *, region: 'Region | None' = None) -> None:
        """Notifica l'arrivo di una richiesta nella regione di ingresso."""
        fields = self._request_fields(request)
        fields['region'] = region
        self.notify(RequestArrived(**fields))

    def notify_request_queued(self, request: Request) -> None:
        """Notifica l'accodamento sul nodo assegnato."""
        node = request.node
        assert node is not None
        self.notify(RequestQueued(**self._request_fields(request), node=node))

    def notify_request_dequeued(self, request: Request) -> None:
        """Notifica l'estrazione dalla FIFO del nodo."""
        node = request.node
        assert node is not None
        self.notify(RequestDequeued(**self._request_fields(request), node=node))

    def notify_request_forwarded(self, request: Request, source: 'Region') -> None:
        """Notifica la partenza verso un'altra regione."""
        self.notify(RequestForwarded(**self._request_fields(request), source_region=source))

    def notify_request_delivered(self, request: Request) -> None:
        """Notifica l'arrivo della richiesta inoltrata a destinazione."""
        self.notify(RequestDelivered(**self._request_fields(request)))

    def notify_request_accepted(self, request: Request, *, queue_wait_s: float | None = None) -> None:
        """Notifica l'ammissione e l'eventuale attesa nella FIFO."""
        self.notify(RequestAccepted(
            **self._request_fields(request),
            node=request.node,
            warm=request.warm,
            queue_wait_s=queue_wait_s,
        ))

    def notify_request_rejected(self, request: Request) -> None:
        """Notifica il rifiuto definitivo di una richiesta."""
        self.notify(RequestRejected(
            **self._request_fields(request),
            reason=request.rejection_reason or 'unknown',
            node=request.node,
        ))

    def notify_request_completed(self, request: Request) -> None:
        """Notifica il completamento di una richiesta e la consegna della risposta."""
        ctx = self._ensure_context()
        self.notify(RequestCompleted(
            **self._request_fields(request),
            node=request.node,
            container=request.container,
            response_time_s=ctx.engine.now - request.arrival_time,
            service_time_s=request.service_duration,
        ))

    ########################################
    ### Node and container notifications ###
    ########################################

    def notify_node_registered(self, node: Node) -> None:
        """Pubblica lo stato iniziale del nodo al momento della registrazione."""
        ctx = self._ensure_context()
        containers = chain.from_iterable(node.containers.values())
        self.notify(NodeRegistered(
            ctx.simulation.run_id,
            ctx.engine.now,
            node.region,
            node.ring,
            node,
            ResourceSnapshot.from_node(node),
            tuple(ContainerSnapshot(container, container.state) for container in containers),
        ))

    def notify_container_state(
        self, node: Node, container: DockerContainer, previous: ContainerState | None, current: ContainerState,
    ) -> None:
        """Pubblica la transizione di stato del container e le prenotazioni risultanti del nodo."""
        ctx = self._ensure_context()
        notification = ContainerStateChanged
        if previous is ContainerState.CREATING and current in (ContainerState.BUSY, ContainerState.IDLE):
            notification = ContainerStarted
        elif previous is ContainerState.IDLE and current is ContainerState.DESTROYED:
            notification = ContainerStopped

        self.notify(notification(
            ctx.simulation.run_id,
            ctx.engine.now,
            node.ring,
            node,
            container,
            region=node.region,
            previous=previous,
            state=current,
            resources=ResourceSnapshot.from_node(node),
        ))


# Observer condiviso a livello di processo per i subscriber esterni (es. logger CLI).
observer = Observer()
