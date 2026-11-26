"""Evento periodico di riconciliazione della cache risorse del load balancer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .event import Event

if TYPE_CHECKING:
    from ..topology.region import Region


@dataclass(frozen=True)
class LoadBalancerRefreshEvent(Event):
    """
    Polling periodico dello stato dei nodi da parte del proxy (load balancer).

    Riproduce il ciclo ``updateTargets`` / ``refreshTargets`` di Serverledge
    (``internal/lb/lb.go``): ogni ``interval_s`` secondi il load balancer
    interroga ``/status`` su tutti i nodi attivi della regione e riallinea la
    propria ``NodeMetricCache`` ai valori reali di CPU e memoria disponibili,
    correggendo eventuali drift dovuti al decremento pessimistico in ``Next()``
    o a container warm/scaduti.
    """

    region: 'Region'
    interval_s: float

    def execute(self) -> None:
        """Riallinea la cache del load balancer della regione e ripianifica il polling."""
        now = self.ctx.engine.now
        infrastructure = self.region.infrastructure
        if infrastructure is None:
            return

        # Aggiorna le metriche del load balancer della regione.
        infrastructure.refresh_load_balancer_metrics(now=now)
        self.ctx.engine.schedule_after(self.interval_s, self)
