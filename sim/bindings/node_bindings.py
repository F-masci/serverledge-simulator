"""Specifica dei binding di ciclo di vita da associare ai nodi."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.model.container import ContainerState, DockerContainer
    from sim.model.node import Node
    from sim.model.request import Request


@dataclass(frozen=True)
class NodeBindings:
    """
    Contenitore immutabile dei riferimenti alle callback di un nodo.

    Viene propagato dalla simulazione fino agli anelli e all'infrastruttura
    così che ogni nodo — sia creato all'avvio sia aggiunto dinamicamente a
    runtime (es. da un autoscaler) — riceva automaticamente i propri binding
    al momento effettivo della creazione.
    """

    ### Lifecycle Callbacks ###

    on_node_registered: Callable[[Node, str, str], None] | None = None

    ### Container Callbacks ###

    on_container_created: Callable[[Node, DockerContainer], None] | None = None
    on_container_acquired: Callable[[Node, DockerContainer], None] | None = None
    on_container_destroyed: Callable[[Node, DockerContainer], None] | None = None
    on_container_state_changed: (
        Callable[[str, str, Node, DockerContainer, ContainerState | None, ContainerState], None] | None
    ) = None
    on_container_idle: Callable[[Node, DockerContainer], None] | None = None
    on_container_busy: Callable[[Node, DockerContainer], None] | None = None

    ### Request Callbacks ###

    on_request_submitted: Callable[[Node, Request], None] | None = None
    on_request_dequeued: Callable[[Node, Request], None] | None = None

    ###############
    ### Binding ###
    ###############

    def bind(self, node: Node) -> None:
        """Associa al nodo tutte le callback configurate e notifica la registrazione."""
        region_name = node.region.name
        ring_id = node.ring.id

        # Collega le callback di ciclo di vita dei container sul nodo.
        if self.on_container_created is not None:
            node.on_container_created = self.on_container_created
        if self.on_container_acquired is not None:
            node.on_container_acquired = self.on_container_acquired
        if self.on_container_destroyed is not None:
            node.on_container_destroyed = self.on_container_destroyed
        if self.on_container_state_changed is not None:
            node.on_container_state_changed = partial(
                self.on_container_state_changed, region_name, ring_id
            )
        if self.on_container_idle is not None:
            node.on_container_idle = self.on_container_idle
        if self.on_container_busy is not None:
            node.on_container_busy = self.on_container_busy

        # Collega le callback di gestione delle richieste sul nodo.
        if self.on_request_submitted is not None:
            node.on_request_submitted = self.on_request_submitted
        if self.on_request_dequeued is not None:
            node.on_request_dequeued = self.on_request_dequeued

        # Notifica la registrazione del nodo dopo aver collegato le sue callback.
        if self.on_node_registered is not None:
            self.on_node_registered(node, region_name, ring_id)
