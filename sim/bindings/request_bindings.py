"""Specifica dei binding di ciclo di vita da associare alle richieste."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.model.request import Request, RequestStatus


@dataclass(frozen=True)
class RequestBindings:
    """
    Contenitore immutabile dei riferimenti alle callback di una richiesta.

    Viene applicato al momento della creazione della richiesta così che tutte
    le transizioni di stato successive (accodamento, ammissione diretta o da
    coda FIFO, consegna remota, rifiuto, completamento) invochino automaticamente
    gli handler configurati senza doverli ripetere sui singoli eventi.
    """

    ### Lifecycle Callbacks ###

    on_queued: Callable[[Request], None] | None = None
    on_accepted: Callable[[Request], None] | None = None
    on_rejected: Callable[[Request], None] | None = None
    on_completed: Callable[[Request], None] | None = None
    on_state_changed: Callable[[Request, RequestStatus, RequestStatus], None] | None = None

    ###############
    ### Binding ###
    ###############

    def bind(self, request: Request) -> None:
        """Associa alla richiesta tutte le callback di ciclo di vita configurate."""
        if self.on_queued is not None:
            request.on_queued = self.on_queued
        if self.on_accepted is not None:
            request.on_accepted = self.on_accepted
        if self.on_rejected is not None:
            request.on_rejected = self.on_rejected
        if self.on_completed is not None:
            request.on_completed = self.on_completed
        if self.on_state_changed is not None:
            request.on_state_changed = self.on_state_changed
