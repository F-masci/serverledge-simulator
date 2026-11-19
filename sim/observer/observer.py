"""Distribuzione sincrona delle notifiche ai subscriber del processo."""

from collections.abc import Callable
import logging
from typing import TYPE_CHECKING

from .notifications import Notification, RequestArrived, RequestAccepted, RequestRejected
from ..model.request import Request
from ..model.node import Node

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext


class _Observer:
    def __init__(self) -> None:
        self._ctx: SimulationContext | None = None
        self._subscribers: dict[object, Callable[[Notification], None]] = {}

    def bind_context(self, ctx: 'SimulationContext') -> None:
        """Associa il contesto della replica senza modificare le iscrizioni."""
        self._ctx = ctx

    def subscribe(self, handler: Callable[[Notification], None]) -> Callable[[], None]:
        """Registra il subscriber e restituisce una funzione per scollegarlo."""
        subscription = object()
        self._subscribers[subscription] = handler

        def unsubscribe() -> None:
            self._subscribers.pop(subscription, None)

        return unsubscribe

    def notify(self, notification: Notification) -> None:
        """Notifica i subscriber in ordine di iscrizione."""
        if not isinstance(notification, Notification):
            raise TypeError('La notifica deve implementare Notification')
        # Una callback può iscriversi o scollegarsi durante la notifica.
        for handler in tuple(self._subscribers.values()):
            try:
                handler(notification)
            except Exception:
                logging.getLogger(__name__).exception('Errore nel subscriber di %s', notification.type)

    def _ensure_context(self) -> 'SimulationContext':
        if self._ctx is None:
            raise RuntimeError('Observer non associato a un contesto di simulazione')
        return self._ctx

    #----------------------------------#
    #   Custom notifications methods   #
    # ----------------------------------#

    ## Arrival Requests ##

    def notify_arrival(self, request: Request):
        """Notifica l'arrivo di una richiesta per una funzione."""
        ctx = self._ensure_context()
        notification = RequestArrived(
            run_id=ctx.simulation.run_id,
            sim_time_s=ctx.engine.now,
            request_id=request.id,
            function=request.function.name
        )
        self.notify(notification)

    def notify_request_accepted(self, request: Request, node: Node):
        """Notifica l'accettazione di una richiesta da parte di un nodo."""
        ctx = self._ensure_context()
        notification = RequestAccepted(
            run_id=ctx.simulation.run_id,
            sim_time_s=ctx.engine.now,
            request_id=request.id,
            function=request.function.name,
            node_id=node.id.key,
        )
        self.notify(notification)

    def notify_request_rejected(self, request: Request):
        """Notifica il rifiuto di una richiesta."""
        ctx = self._ensure_context()
        notification = RequestRejected(
            run_id=ctx.simulation.run_id,
            sim_time_s=ctx.engine.now,
            request_id=request.id,
            function=request.function.name,
            reason=request.rejection_reason or 'unknown',
            node_id=request.node.id.key if request.node is not None else None,
        )
        self.notify(notification)

observer = _Observer()
