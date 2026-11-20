"""Distribuzione sincrona delle notifiche della replica e del processo."""

from collections.abc import Callable
import logging
from itertools import chain
from typing import TYPE_CHECKING

from .notifications import (
    ContainerStarted, ContainerStateChanged, ContainerStopped, NodeRegistered, Notification,
    RequestAccepted, RequestArrived, RequestCompleted, RequestDelivered, RequestForwarded, RequestQueued,
    RequestRejected,
)
from .notifications.node import ContainerSnapshot, ResourceSnapshot
from ..model.container import ContainerState, DockerContainer
from ..model.node import Node
from ..model.request import Request

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext


class _Observer:
    def __init__(
        self, ctx: 'SimulationContext | None' = None, *, parent: '_Observer | None' = None,
    ) -> None:
        self._ctx = ctx
        self._parent = parent
        self._node_rings: dict[Node, str] = {}
        self._subscribers: dict[object, tuple[Callable[[Notification], None], bool]] = {}

    def subscribe(
        self, handler: Callable[[Notification], None], *, propagate_errors: bool = False,
    ) -> Callable[[], None]:
        """Registra il subscriber e restituisce una funzione per scollegarlo."""
        subscription = object()
        self._subscribers[subscription] = (handler, propagate_errors)

        def unsubscribe() -> None:
            self._subscribers.pop(subscription, None)

        return unsubscribe

    def notify(self, notification: Notification) -> None:
        """Notifica i subscriber della replica, poi quelli del processo."""
        if not isinstance(notification, Notification):
            raise TypeError('La notifica deve implementare Notification')
        # Una callback può iscriversi o scollegarsi durante la notifica.
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
        if self._ctx is None:
            raise RuntimeError('Observer non associato a un contesto di simulazione')
        return self._ctx

    def _request_fields(self, request: Request) -> dict:
        ctx = self._ensure_context()
        node = request.node
        region = request.region
        if node is not None and region is not None and node not in self._node_rings:
            if region.infrastructure is not None:
                for ring_id, ring in region.infrastructure.rings.items():
                    if node in ring:
                        self.notify_node_registered(node, region.name, ring_id)
                        break
        return {
            'run_id': ctx.simulation.run_id,
            'sim_time_s': ctx.engine.now,
            'request_id': request.id,
            'function': request.function.name,
            'region': region.name if region is not None else None,
            'ring_id': self._node_rings.get(node),
        }

    def notify_arrival(self, request: Request, *, region: str | None = None) -> None:
        """Notifica l'arrivo di una richiesta per una funzione."""
        fields = self._request_fields(request)
        fields['region'] = region
        self.notify(RequestArrived(**fields))

    def notify_request_queued(self, request: Request) -> None:
        """Notifica l'accodamento sul nodo assegnato."""
        node = request.node
        assert node is not None
        self.notify(RequestQueued(**self._request_fields(request), node_id=node.id.key))

    def notify_request_forwarded(self, request: Request, source: str) -> None:
        """Notifica la partenza verso un'altra regione."""
        self.notify(RequestForwarded(**self._request_fields(request), source_region=source))

    def notify_request_delivered(self, request: Request) -> None:
        """Notifica l'arrivo della richiesta inoltrata a destinazione."""
        self.notify(RequestDelivered(**self._request_fields(request)))

    def notify_request_accepted(self, request: Request, *, queue_wait_s: float | None = None) -> None:
        """Notifica l'ammissione e l'eventuale attesa nella FIFO."""
        node = request.node
        self.notify(RequestAccepted(
            **self._request_fields(request),
            node_id=node.id.key if node is not None else None,
            warm=request.warm,
            queue_wait_s=queue_wait_s,
        ))

    def notify_request_rejected(self, request: Request) -> None:
        """Notifica il rifiuto di una richiesta."""
        node = request.node
        self.notify(RequestRejected(
            **self._request_fields(request),
            reason=request.rejection_reason or 'unknown',
            node_id=node.id.key if node is not None else None,
        ))

    def notify_request_completed(self, request: Request) -> None:
        """Notifica il completamento di una richiesta."""
        ctx = self._ensure_context()
        node, container = request.node, request.container
        self.notify(RequestCompleted(
            **self._request_fields(request),
            node_id=node.id.key if node is not None else None,
            container_id=container.id if container is not None else None,
            response_time_s=ctx.engine.now - request.arrival_time,
            service_time_s=request.service_duration,
        ))

    def notify_node_registered(self, node: Node, region: str, ring_id: str) -> None:
        """Pubblica lo stato iniziale del nodo."""
        ctx = self._ensure_context()
        self._node_rings[node] = ring_id
        containers = chain.from_iterable(node.containers.values())
        self.notify(NodeRegistered(
            ctx.simulation.run_id, ctx.engine.now, region, ring_id, node.id.key,
            ResourceSnapshot.from_node(node),
            tuple(ContainerSnapshot(c.id, c.function.name, c.state) for c in containers),
        ))

    def notify_container_state(
        self, region: str, ring_id: str, node: Node, container: DockerContainer, previous: ContainerState | None,
    ) -> None:
        """Pubblica la transizione e le prenotazioni risultanti."""
        ctx = self._ensure_context()
        notification = ContainerStateChanged
        if previous is ContainerState.CREATING and container.state in (ContainerState.BUSY, ContainerState.IDLE):
            notification = ContainerStarted
        elif previous is ContainerState.IDLE and container.state is ContainerState.DESTROYED:
            notification = ContainerStopped
        self.notify(notification(
            ctx.simulation.run_id, ctx.engine.now, ring_id, node.id.key, container.id, container.function.name,
            region=region, previous=previous, state=container.state, resources=ResourceSnapshot.from_node(node),
        ))


observer = _Observer()
