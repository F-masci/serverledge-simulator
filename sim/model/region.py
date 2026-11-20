"""Ingresso geografico delle invocazioni."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from .infrastructure import Infrastructure
from .node import Node
from .request import Request


@dataclass(frozen=True)
class Region:
    name: str
    infrastructure: Infrastructure | None
    service_time_s: float = 0.0
    offload_latency_s: float = 0.0
    offload_latency_by_source_s: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))

    def offload_latency_from(self, source: str) -> float:
        """Ritardo totale di andata e ritorno dalla regione sorgente."""
        return self.offload_latency_by_source_s.get(source, self.offload_latency_s)

    @property
    def mode(self) -> str:
        return 'simulated' if self.infrastructure is not None else 'always_succeeds'

    def submit(
        self, request: Request, *, now: float, node: Node | None = None, area: str | None = None,
    ) -> Node | None:
        request.enter_region(self.name, now)
        if self.infrastructure is not None:
            return self.infrastructure.submit(request, now=now, node=node, area=area)

        request.accept_without_container(self.service_time_s)
        return None
