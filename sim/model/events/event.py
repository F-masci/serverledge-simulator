"""Contratto comune degli eventi di simulazione."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from ..request import Request

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext


@dataclass(frozen=True, kw_only=True)
class Event(ABC):
    """
    Contratto comune degli eventi di simulazione.

    Un evento di simulazione rappresenta un'azione che modifica lo stato del modello
    e può pianificare ulteriori eventi.
    """

    ctx: 'SimulationContext' = field(repr=False, compare=False)

    @abstractmethod
    def execute(self) -> None:
        """Modifica il modello e può pianificare ulteriori eventi."""

@dataclass(frozen=True, kw_only=True)
class RequestEvent(Event):
    """
    Evento di simulazione associato a una richiesta.

    Questo evento rappresenta un'azione che coinvolge una richiesta specifica
    e può modificare lo stato della richiesta o pianificare ulteriori eventi
    correlati alla richiesta.
    """

    ### Callback ###
    on_request_submitted: Callable[[Request], None] | None = None
    on_request_accepted: Callable[[Request], None] | None = None
    on_request_rejected: Callable[[Request], None] | None = None
    on_request_queued: Callable[[Request], None] | None = None

    @abstractmethod
    def execute(self) -> None:
        """Modifica lo stato della richiesta e può pianificare ulteriori eventi."""

    ## Callback handling ##

    def _handle_callback(self, request: Request) -> None:
        """Invoca il callback corrispondente allo stato della richiesta."""
        if self.on_request_submitted is not None:
            self.on_request_submitted(request)

        if request.is_accepted and self.on_request_accepted is not None:
            self.on_request_accepted(request)
        elif request.is_rejected and self.on_request_rejected is not None:
            self.on_request_rejected(request)
        elif request.is_queued and self.on_request_queued is not None:
            self.on_request_queued(request)