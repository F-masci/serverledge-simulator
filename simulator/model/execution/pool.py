"""Pool locale di container di una singola funzione.

Le modifiche sono riservate al nodo, che coordina la contabilità delle risorse.
"""

import json
from collections.abc import Iterator

from .container import DockerContainer
from .function import Function


class ContainerPool:
    """Pool locale di container di una singola funzione."""

    def __init__(self, function: Function) -> None:
        self._function = function
        self._busy: list[DockerContainer] = []
        self._idle: list[DockerContainer] = []

    @property
    def function(self) -> Function:
        return self._function

    @property
    def busy(self) -> tuple[DockerContainer, ...]:
        return tuple(self._busy)

    @property
    def idle(self) -> tuple[DockerContainer, ...]:
        return tuple(self._idle)

    #################################
    ### Internal state management ###
    #################################

    ## Container selection ##

    def get_reusable_container(self) -> DockerContainer | None:
        """Restituisce un container pronto per l'uso, se disponibile."""

        # Prima si cerca un container già in uso che
        # non abbia raggiunto la concorrenza massima.
        busy_container = self._get_reusable_busy()
        if busy_container is not None:
            return busy_container

        # Poi si cerca un container idle che
        # non abbia raggiunto la concorrenza massima.
        idle_container = self._get_reusable_idle()
        if idle_container is not None:
            return idle_container

        return None

    def _get_reusable_busy(self) -> DockerContainer | None:
        """Restituisce un container già in uso che non abbia raggiunto la concorrenza massima, se disponibile."""
        return next((c for c in self._busy
                     if c.requests_count < self.function.max_concurrency), None)

    def _get_reusable_idle(self) -> DockerContainer | None:
        """Restituisce l'ultimo container idle pronto per l'uso, se disponibile."""
        return self._idle[-1] if self._idle else None

    def pop_idle(self) -> DockerContainer:
        """Estrae l'ultimo container idle secondo politica LIFO."""
        return self._idle.pop()

    ## Container management ##

    def add_busy(self, container: DockerContainer) -> None:
        """Registra un container nel gruppo busy."""
        self._ensure_function(container)
        self._ensure_duplicate(container)
        self._busy.append(container)

    def add_idle(self, container: DockerContainer) -> None:
        """Registra un container nel gruppo idle."""
        self._ensure_function(container)
        self._ensure_duplicate(container)
        self._idle.append(container)

    def add(self, container: DockerContainer) -> None:
        """Registra un container nel pool, in base al suo stato attuale."""
        if container.is_busy:
            self.add_busy(container)
            return

        if container.is_idle:
            self.add_idle(container)
            return

        raise ValueError('Container non pronto')

    @staticmethod
    def _swap_remove(items: list[DockerContainer], container: DockerContainer) -> None:
        """Rimuove un container dalla lista senza preservare l'ordine."""
        index = items.index(container)
        items[index] = items[-1]
        items.pop()

    def remove_busy(self, container: DockerContainer) -> None:
        """Rimuove un container occupato dal pool senza preservare l'ordine."""
        self._swap_remove(self._busy, container)

    def remove_idle(self, container: DockerContainer, *, preserve_order: bool = False) -> None:
        """Rimuove un container idle dal pool (con compattazione ordinata per il janitor)."""
        if preserve_order:
            self._idle.remove(container)
        else:
            self._swap_remove(self._idle, container)

    ######################
    ### Error handling ###
    ######################

    def _ensure_function(self, container: DockerContainer) -> None:
        if container.function != self.function:
            raise ValueError('Container incompatibile con il pool')

    def _ensure_duplicate(self, container: DockerContainer) -> None:
        if container in self:
            raise ValueError('Container già presente nel pool')

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una rappresentazione testuale del pool."""
        return f'ContainerPool(function={self.function.name}, busy={len(self._busy)}, idle={len(self._idle)})'

    def to_dict(self) -> dict:
        """Restituisce la funzione e lo stato dei container nei due gruppi del pool."""
        return {
            'function': self.function.name,
            'busy': [container.to_dict() for container in self._busy],
            'idle': [container.to_dict() for container in self._idle],
        }

    def to_json(self) -> str:
        """Serializza il pool per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    ###############
    ### Utility ###
    ###############

    def __len__(self) -> int:
        """Restituisce il numero totale di container nel pool."""
        return len(self._busy) + len(self._idle)

    def __iter__(self) -> Iterator[DockerContainer]:
        """Restituisce un iteratore sui container del pool."""
        return iter(self.busy + self.idle)

    def __contains__(self, container: object) -> bool:
        """Restituisce True se il container è presente nel pool, False altrimenti."""
        if not isinstance(container, DockerContainer):
            return False
        return container in self._busy or container in self._idle
