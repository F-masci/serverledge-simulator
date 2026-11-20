"""Ingresso geografico delle invocazioni."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .infrastructure import Infrastructure
from .node import Node
from .request import Request


@dataclass(frozen=True, init=False)
class Region:
    name: str
    infrastructure: Infrastructure | None
    service_time_s: float
    offload_latency_s: float
    offload_latency_by_source_s: Mapping[str, float]
    has_load_balancer: bool

    def __init__(
        self, name: str, *, build_infrastructure: Callable[['Region'], Infrastructure] | None = None,
        service_time_s: float = 0.0, offload_latency_s: float = 0.0,
        offload_latency_by_source_s: Mapping[str, float] | None = None,
        has_load_balancer: bool = True,
    ) -> None:
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'service_time_s', service_time_s)
        object.__setattr__(self, 'offload_latency_s', offload_latency_s)
        object.__setattr__(self, 'offload_latency_by_source_s', MappingProxyType(dict(offload_latency_by_source_s or {})))
        object.__setattr__(self, 'has_load_balancer', has_load_balancer)
        infrastructure = build_infrastructure(self) if build_infrastructure is not None else None
        object.__setattr__(self, 'infrastructure', infrastructure)

    def offload_latency_from(self, source: str) -> float:
        """Ritardo totale di andata e ritorno dalla regione sorgente."""
        return self.offload_latency_by_source_s.get(source, self.offload_latency_s)

    def first_registered_node(self) -> Node | None:
        if self.infrastructure is None:
            return None
        # Come nel registry Go, ordina per chiave machine tag/nome.
        return min((node for ring in self.infrastructure.rings.values() for node in ring),
                   key=lambda node: f'{node.id.machine_tag}/{node.id.key}', default=None)

    @property
    def mode(self) -> str:
        return 'simulated' if self.infrastructure is not None else 'always_succeeds'

    def submit(
        self, request: Request, *, now: float, node: Node | None = None,
    ) -> Node | None:
        request.enter_region(self, now)
        if self.infrastructure is not None:
            if node is None and not self.has_load_balancer:
                node = self.first_registered_node()
                if node is None:
                    request.reject('no_destination')
                    return None
            return self.infrastructure.submit(request, now=now, node=node)

        request.accept_on_region(self)
        return None
