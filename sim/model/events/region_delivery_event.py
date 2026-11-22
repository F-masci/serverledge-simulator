"""Consegna di un'invocazione inoltrata a un'altra regione."""

from dataclasses import dataclass

from .event import Event
from ..node import Node
from ..region import Region
from ..request import Request


@dataclass(frozen=True)
class RegionDeliveryEvent(Event):
    request: Request
    region: Region
    node: Node | None = None

    def execute(self) -> None:
        """Inoltra la richiesta alla regione di destinazione."""
        now = self.ctx.now

        # Notifica l'osservatore della consegna della richiesta alla regione di destinazione.
        self.ctx.observer.notify_request_delivered(self.request)

        # Inoltra la richiesta alla regione di destinazione (le callback sono già legate alla richiesta).
        self.region.submit(self.request, now=now, node=self.node)
