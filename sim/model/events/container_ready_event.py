"""Fine della creazione di un container riservato a una richiesta."""

from dataclasses import dataclass

from ..request import Request
from .completion_event import CompletionEvent
from .event import Event

@dataclass(frozen=True)
class ContainerReadyEvent(Event):
    request: Request
    arrival_time: float

    def execute(self) -> None:
        node, container = self.request.node, self.request.container
        if node is None or container is None:
            raise ValueError('La richiesta non ha un container assegnato')
        duration = self.request.service_duration
        if duration is None:
            raise ValueError('La richiesta deve avere una durata di servizio')
        engine = self.ctx.engine
        node.creation_succeeded(container, engine.now)
        engine.schedule_after(duration, CompletionEvent(self.request, self.arrival_time, ctx=self.ctx))
