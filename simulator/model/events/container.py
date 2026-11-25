"""Eventi del ciclo di vita dei container: cold start completato e pulizia idle."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .event import Event
from .request import CompletionEvent
from ..execution.request import Request

if TYPE_CHECKING:
    from ..execution.container import DockerContainer
    from ..topology.node import Node


@dataclass(frozen=True)
class ContainerReadyEvent(Event):
    """Evento di completamento della creazione (cold start) di un container."""

    request: Request

    def execute(self) -> None:
        """Segnala la fine della creazione del container e pianifica il completamento del servizio."""
        engine = self.ctx.engine

        node, container = self.request.node, self.request.container
        if node is None or container is None:
            raise ValueError('La richiesta non ha un container assegnato')

        duration = self.request.service_duration
        if duration is None:
            raise ValueError('La richiesta deve avere una durata di servizio')

        # Notifica il nodo della riuscita della creazione del container.
        node.creation_succeeded(container, engine.now)

        # Pianifica il completamento del servizio.
        completion_event = CompletionEvent(ctx=self.ctx, request=self.request)
        engine.schedule_after(duration, completion_event)


@dataclass(frozen=True)
class ContainerCleanupEvent(Event):
    """Evento periodico del janitor per verificare la scadenza di un container idle."""

    node: 'Node'
    container: 'DockerContainer'

    def execute(self) -> None:
        """Segnala al nodo di rimuovere il container idle."""
        self.node.expire(self.ctx.engine.now, container=self.container)
