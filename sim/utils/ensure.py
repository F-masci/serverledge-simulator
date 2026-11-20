from sim.model.node import Node
from sim.model.request import Request

def ensure_node(request: Request) -> Node:
    node = request.node
    if node is None:
        raise ValueError('La richiesta non è stata assegnata a un nodo')
    return node

def ensure_container(request: Request):
    container = request.container
    if container is None:
        raise ValueError('La richiesta non è stata assegnata a un container')
    return container