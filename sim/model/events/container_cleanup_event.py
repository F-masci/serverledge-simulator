"""Controllo del janitor su un container idle."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .event import Event

if TYPE_CHECKING:
    from ..container import DockerContainer
    from ..node import Node


@dataclass(frozen=True)
class ContainerCleanupEvent(Event):
    node: 'Node'
    container: 'DockerContainer'

    def execute(self) -> None:
        self.node.expire(self.ctx.engine.now, container=self.container)
