"""Politiche di scheduling delle invocazioni di Serverledge."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from itertools import islice
from typing import TYPE_CHECKING

from .request import Request

if TYPE_CHECKING:
    from .node import Node
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

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile della configurazione di scheduling."""
        fields = self.to_dict()
        details = ', '.join(f'{name}={value}' for name, value in fields.items() if value is not None)
        return f'NodeScheduling({details})'

    def to_dict(self) -> dict:
        """Restituisce la configurazione di scheduling con i riferimenti espressi come ID."""
        return {
            'policy': self.policy.value,
            'fallback_local': self.fallback_local,
            'nearest_neighbors': (
                [node.id.key for node in self.nearest_neighbors]
                if self.nearest_neighbors is not None
                else None
            ),
            'remote_region': self.remote_region.name if self.remote_region is not None else None,
        }

    def to_json(self) -> str:
        """Serializza la configurazione di scheduling per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        remote = self.remote_region.name if self.remote_region is not None else None
        return (
            f'<NodeScheduling policy={self.policy.value} '
            f'fallback_local={self.fallback_local} remote_region={remote}>'
        )



class OffloadingScheduler(ABC):
    """Ammette, rifiuta o individua la destinazione di una richiesta."""

    def __init__(self, nodes: Callable[[], tuple[Node, ...]], settings: NodeScheduling) -> None:
        # La funzione nodes() restituisce la lista dei nodi registrati nella regione del nodo.
        self._nodes = nodes
        self._settings = settings

    @staticmethod
    def create(nodes: Callable[[], tuple[Node, ...]], settings: NodeScheduling) -> 'OffloadingScheduler':
        """Costruisce lo scheduler della policy configurata sul nodo."""
        return _SCHEDULER_TYPES[settings.policy](nodes, settings)

    @abstractmethod
    def schedule(self, request: Request, node: Node) -> 'Node | Region | None':
        """Restituisce un destinatario per l'inoltro o None dopo una decisione locale."""

    @staticmethod
    def _submit_locally(request: Request, node: Node, *, allow_queue: bool, reject_on_failure: bool = True) -> bool:
        """Tenta l'ammissione locale e restituisce l'esito del tentativo."""

        # Verifica la compatibilità dell'architettura prima dell'ammissione.
        if not request.function.supports_arch(node.id.arch):
            if reject_on_failure:
                request.reject('out_of_resources', node=node)
            return False

        # Inoltra la richiesta al nodo con la politica di accodamento configurata.
        return node.submit(request, allow_queue=allow_queue, reject_on_failure=reject_on_failure)

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile dello scheduler."""
        fields = self._settings.to_dict()
        details = ', '.join(f'{name}={value}' for name, value in fields.items() if value is not None)
        return f'{type(self).__name__}({details})'

    def to_dict(self) -> dict:
        """Restituisce lo stato dello scheduler in un formato serializzabile."""
        return {
            'type': type(self).__name__,
            **self._settings.to_dict(),
        }

    def to_json(self) -> str:
        """Serializza lo scheduler per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<{type(self).__name__} policy={self._settings.policy.value}>'


class DefaultScheduler(OffloadingScheduler):
    """Esegue le invocazioni sul nodo di ingresso con la sua FIFO locale."""

    def schedule(self, request: Request, node: Node) -> None:
        """Consegna la richiesta al nodo se l'architettura è compatibile."""
        self._submit_locally(request, node, allow_queue=True)


class EdgeOnlyScheduler(OffloadingScheduler):
    """Delega l'esecuzione ai vicini della stessa regione."""

    def schedule(self, request: Request, node: Node) -> Node | None:
        """Sceglie un vicino o tenta l'esecuzione locale secondo il fallback configurato."""

        # Se l'invocazione può essere inoltrata, cerca un vicino compatibile con più memoria disponibile.
        if request.can_offload:
            target = self._edge_target(request, node, self._settings.nearest_neighbors)
            if target is not None:
                return target

            # Se il fallback locale non è abilitato, rifiuta la richiesta.
            if not self._settings.fallback_local:
                request.reject('out_of_resources', node=node)
                return None

        # Prova l'esecuzione locale senza accodare la richiesta.
        self._submit_locally(request, node, allow_queue=False)

        return None

    def _edge_target(self, request: Request, node: Node, neighbors: tuple[Node, ...] | None) -> Node | None:
        """Sceglie il vicino compatibile con più memoria disponibile al momento."""
        # Vengono estratti tutti i nodi registrati nella regione del nodo, escludendo il nodo stesso.
        # Se sono specificati vicini, vengono filtrati tra quelli registrati.
        #
        # TODO: modellare lo snapshot periodico del registry e il ritardo di scoperta.
        nodes = self._nodes()
        peers = (peer for peer in nodes if peer is not node) # Esclude il nodo stesso dai candidati.

        # Se non sono specificati vicini, considera i primi due peer registrati come candidati.
        if neighbors is None:
            candidates = tuple(islice(peers, 2))
        else:
            registered = set(peers)
            candidates = tuple(peer for peer in neighbors if peer in registered)

        # Filtra i candidati per trovare quelli compatibili con la funzione richiesta.
        suitable = (
            peer for peer in candidates
            if peer.resources.available_memory > 0 and request.function.supports_arch(peer.id.arch)
        )

        # Restituisce il vicino con più memoria disponibile, o None se non ci sono candidati idonei.
        return max(suitable, key=lambda peer: peer.resources.available_memory, default=None)


class CloudOnlyScheduler(OffloadingScheduler):
    """Delega l'esecuzione alla regione remota configurata sul nodo."""

    def schedule(self, request: Request, node: Node) -> 'Region | None':
        """Restituisce la regione remota se l'inoltro è consentito, altrimenti rifiuta."""

        # Se l'invocazione non può essere inoltrata, rifiuta la richiesta.
        if not request.can_offload:
            request.reject('offloading_not_allowed', node=node)
            return None

        # Se non è configurata una regione remota, rifiuta la richiesta.
        if self._settings.remote_region is None:
            request.reject('no_remote_region', node=node)
            return None

        # Restituisce la regione remota configurata per l'inoltro della richiesta.
        return self._settings.remote_region


class EdgeCloudScheduler(CloudOnlyScheduler):
    """Dà priorità all'esecuzione locale e usa la regione remota come fallback."""

    def schedule(self, request: Request, node: Node) -> 'Region | None':
        """Tenta l'ammissione locale immediata e applica la policy cloud se fallisce."""

        # Se l'invocazione può essere eseguita localmente, tenta l'ammissione senza accodare la richiesta.
        if self._submit_locally(request, node, allow_queue=False, reject_on_failure=False):
            return None

        # Se l'ammissione locale fallisce, applica la policy cloud per l'inoltro.
        return super().schedule(request, node)

_SCHEDULER_TYPES: dict[SchedulingPolicy, type[OffloadingScheduler]] = {
    SchedulingPolicy.DEFAULT: DefaultScheduler,
    SchedulingPolicy.EDGE_ONLY: EdgeOnlyScheduler,
    SchedulingPolicy.EDGE_CLOUD: EdgeCloudScheduler,
    SchedulingPolicy.CLOUD_ONLY: CloudOnlyScheduler,
}
