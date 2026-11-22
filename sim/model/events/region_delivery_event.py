"""Consegna di un'invocazione inoltrata a un'altra regione."""

from dataclasses import dataclass

from .event import RequestEvent
from ..node import Node
from ..region import Region
from ..request import Request


@dataclass(frozen=True)
class RegionDeliveryEvent(RequestEvent):
    request: Request
    region: Region
    node: Node | None = None

    def execute(self) -> None:
        """Inoltra la richiesta alla regione di destinazione."""
        now = self.ctx.now

        # Notifica l'osservatore della consegna della richiesta alla regione di destinazione.
        self.ctx.observer.notify_request_delivered(self.request)

        # Inoltra la richiesta alla regione di destinazione.
        self.region.submit(self.request, now=now, node=self.node)
        self._handle_callback(self.request)
