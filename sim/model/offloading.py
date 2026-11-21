"""Politiche di scheduling delle invocazioni di Serverledge."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import islice
from typing import TYPE_CHECKING

from .node import Node
from .request import Request

if TYPE_CHECKING:
    from .region import Region


class SchedulingPolicy(StrEnum):
    """Identificatori delle policy selezionabili nello scenario."""

    # Esegue sul nodo locale se compatibile e usa la FIFO se configurata.
    DEFAULT = 'default'
    # Se consentito, inoltra al vicino compatibile con più RAM disponibile.
    # Prova in locale se l'inoltro è disabilitato o se manca un vicino e il fallback è abilitato.
    EDGE_ONLY = 'edgeonly'
    # Prova in locale. Se l'ammissione fallisce, tenta l'inoltro remoto se consentito.
    EDGE_CLOUD = 'edgecloud'
    # Tenta direttamente l'inoltro remoto se consentito, altrimenti rifiuta.
    CLOUD_ONLY = 'cloudonly'

    @classmethod
    def _missing_(cls, value: object) -> 'SchedulingPolicy | None':
        # CreateSchedulingPolicy in Go ricade sulla policy locale per nomi sconosciuti.
        return cls.DEFAULT if isinstance(value, str) else None


@dataclass(frozen=True)
class NodeScheduling:
    """Configurazione della policy di un nodo e delle sue destinazioni di offloading."""

    policy: SchedulingPolicy = SchedulingPolicy.DEFAULT
    fallback_local: bool = False
    nearest_neighbors: tuple[Node, ...] | None = None
    remote_region: 'Region | None' = None



class OffloadingScheduler(ABC):
    """Ammette, rifiuta o individua la destinazione di una richiesta."""

    def __init__(self, nodes: Callable[[], tuple[Node, ...]], settings: NodeScheduling) -> None:
        self._nodes = nodes
        self._settings = settings

    @staticmethod
    def create(nodes: Callable[[], tuple[Node, ...]], settings: NodeScheduling) -> 'OffloadingScheduler':
        """Costruisce lo scheduler della policy configurata sul nodo."""
        return _SCHEDULER_TYPES[settings.policy](nodes, settings)

    @abstractmethod
    def schedule(self, request: Request, node: Node) -> 'Node | Region | None':
        """Restituisce un destinatario per l'inoltro o None dopo una decisione locale."""


class DefaultScheduler(OffloadingScheduler):
    """Esegue le invocazioni sul nodo di ingresso con la sua FIFO locale."""

    def schedule(self, request: Request, node: Node) -> None:
        """Consegna la richiesta al nodo se l'architettura è compatibile."""
        if request.function.supports_arch(node.id.arch):
            node.submit(request, allow_queue=True)
        else:
            request.reject('out_of_resources', node=node)


class EdgeOnlyScheduler(OffloadingScheduler):
    """Delega l'esecuzione ai vicini della stessa regione."""

    def schedule(self, request: Request, node: Node) -> Node | None:
        """Sceglie un vicino o tenta l'esecuzione locale secondo il fallback configurato."""
        if request.can_offload:
            target = self._edge_target(request, node, self._settings.nearest_neighbors)
            if target is not None:
                return target
            if not self._settings.fallback_local:
                request.reject('out_of_resources', node=node)
                return None
        if not request.function.supports_arch(node.id.arch) or not node.submit(request, allow_queue=False, reject_on_failure=False):
            request.reject('out_of_resources', node=node)
        return None

    def _edge_target(self, request: Request, node: Node, neighbors: tuple[Node, ...] | None) -> Node | None:
        """Sceglie il vicino compatibile con più memoria disponibile al momento."""
        # TODO: modellare lo snapshot periodico del registry e il ritardo di scoperta.
        nodes = self._nodes()
        peers = (peer for peer in nodes if peer is not node)
        if neighbors is None:
            candidates = tuple(islice(peers, 2))
        else:
            registered = set(peers)
            candidates = tuple(peer for peer in neighbors if peer in registered)
        suitable = (
            peer for peer in candidates
            if peer.resources.available_memory > 0 and request.function.supports_arch(peer.id.arch)
        )
        return max(suitable, key=lambda peer: peer.resources.available_memory, default=None)


class CloudOnlyScheduler(OffloadingScheduler):
    """Delega l'esecuzione alla regione remota configurata sul nodo."""

    def schedule(self, request: Request, node: Node) -> 'Region | None':
        """Restituisce la regione remota se l'inoltro è consentito, altrimenti rifiuta."""
        if request.can_offload and self._settings.remote_region is not None:
            return self._settings.remote_region
        request.reject('out_of_resources', node=node)
        return None


class EdgeCloudScheduler(CloudOnlyScheduler):
    """Dà priorità all'esecuzione locale e usa la regione remota come fallback."""

    def schedule(self, request: Request, node: Node) -> 'Region | None':
        """Tenta l'ammissione locale immediata e applica la policy cloud se fallisce."""
        if request.function.supports_arch(node.id.arch) and node.submit(request, allow_queue=False, reject_on_failure=False):
            return None
        return super().schedule(request, node)

_SCHEDULER_TYPES: dict[SchedulingPolicy, type[OffloadingScheduler]] = {
    SchedulingPolicy.DEFAULT: DefaultScheduler,
    SchedulingPolicy.EDGE_ONLY: EdgeOnlyScheduler,
    SchedulingPolicy.EDGE_CLOUD: EdgeCloudScheduler,
    SchedulingPolicy.CLOUD_ONLY: CloudOnlyScheduler,
}
