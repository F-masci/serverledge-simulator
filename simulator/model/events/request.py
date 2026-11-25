"""Eventi del ciclo di vita delle richieste: arrivi, consegne, completamenti e risposte."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .event import Event
from ..execution.function import Function
from ..execution.request import Request

if TYPE_CHECKING:
    from simulator.bindings import RequestBindings
    from ..topology.node import Node
    from ..topology.region import Region


@dataclass(frozen=True)
class ArrivalEvent(Event):
    """Evento di arrivo esterno di una nuova invocazione."""

    function: Function
    entry_region: Region
    entry_node: 'Node | None'
    request_bindings: 'RequestBindings | None' = None

    def execute(self) -> None:
        """Pianifica il prossimo arrivo e invia la richiesta alla regione di ingresso."""
        engine = self.ctx.engine
        sampler = self.function.sampler
        if sampler is None:
            raise RuntimeError("La funzione dell'arrivo deve avere un campionatore")

        # TODO: campionare servizio e cold start dopo il placement, da profili
        # calibrati per funzione e destinazione (architettura, capacità e carico).
        # Calibrare anche la latenza di offloading per tratta con misure del reale.
        arrival_time = engine.now
        service_duration = sampler.service.sample()
        can_offload = self.ctx.scenario.invocation.can_offload

        # Crea la richiesta con i parametri campionati e applica subito i suoi binding.
        request = Request(
            self.function, service_duration=service_duration, arrival_time=arrival_time,
            can_offload=can_offload,
        )
        if self.request_bindings is not None:
            self.request_bindings.bind(request)

        # Notifica l'arrivo della richiesta per la funzione nella regione di ingresso.
        self.ctx.observer.notify_arrival(request, region=self.entry_region)

        # Pianifica il prossimo arrivo secondo l'interarrivo campionato.
        next_time = engine.now + sampler.arrivals.next_iat()
        if next_time <= engine.now:
            raise ValueError('Interarrivo troppo piccolo per avanzare il tempo simulato')
        engine.schedule(next_time, self)

        # Inoltra la richiesta nella regione di ingresso.
        self.entry_region.submit(request, now=engine.now, node=self.entry_node)


@dataclass(frozen=True)
class RegionDeliveryEvent(Event):
    """Evento di consegna di una richiesta inoltrata a una regione remota."""

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


@dataclass(frozen=True)
class CompletionEvent(Event):
    """Evento di fine servizio di una richiesta accettata."""

    request: Request

    def execute(self) -> None:
        """Termina il servizio e pianifica l'eventuale ritorno della risposta."""
        now = self.ctx.engine.now
        request = self.request
        if not request.is_accepted:
            raise ValueError('Solo una richiesta accettata può essere completata')

        # Se la richiesta ha un ritardo di ritorno,
        # pianifica l'evento di risposta dopo il ritardo.
        if request.return_delay > 0:
            request.begin_response(now)
            self.ctx.engine.schedule_after(request.return_delay, ResponseEvent(ctx=self.ctx, request=request))
            return

        # Se la richiesta non ha un ritardo di ritorno, completa immediatamente la richiesta.
        # La notifica di completamento viene invocata dalla callback on_completed della richiesta.
        request.complete(now)


@dataclass(frozen=True)
class ResponseEvent(Event):
    """Evento di consegna al client della risposta (o del rifiuto) da una regione remota."""

    request: Request

    def execute(self) -> None:
        """Completa la risposta o il rifiuto della richiesta."""
        request = self.request
        if not request.is_responding:
            raise ValueError('La richiesta non ha una risposta in transito')

        # Distingue un rifiuto remoto in transito da un completamento ordinario.
        if request.rejection_reason is not None:
            request.finish_rejection()
            return

        # Completa la risposta della richiesta; on_completed notifica l'osservatore.
        request.finish_response()
