"""Aggiornamento periodico delle informazioni di offloading."""

from collections.abc import Callable
from dataclasses import dataclass

from .event import Event


@dataclass(frozen=True)
class RegistryRefreshEvent(Event):
    refresh: Callable[[], None]
    interval: int

    def execute(self) -> None:
        self.refresh()
        self.ctx.engine.schedule_after(self.interval, self)
