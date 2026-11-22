"""Controlli di precondizione sui riferimenti associati a una richiesta."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.model.container import DockerContainer
    from sim.model.node import Node
    from sim.model.request import Request


def ensure_node(request: 'Request') -> 'Node':
    """Restituisce il nodo assegnato alla richiesta o solleva ValueError se assente."""
    node = request.node
    if node is None:
        raise ValueError('La richiesta non è stata assegnata a un nodo')
    return node


def ensure_container(request: 'Request') -> 'DockerContainer':
    """Restituisce il container assegnato alla richiesta o solleva ValueError se assente."""
    container = request.container
    if container is None:
        raise ValueError('La richiesta non è stata assegnata a un container')
    return container
