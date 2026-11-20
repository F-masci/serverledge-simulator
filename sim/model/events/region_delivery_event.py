"""Consegna di un'invocazione inoltrata a un'altra regione."""

from dataclasses import dataclass

from ..region import Region
from ..request import Request
from .event import Event


@dataclass(frozen=True)
class RegionDeliveryEvent(Event):
    request: Request
    region: Region

    def execute(self) -> None:
        self.ctx.simulation.deliver_region(self.request, self.region)
