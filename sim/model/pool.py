"""Pool locale di container di una singola funzione.

Le modifiche sono riservate al nodo, che coordina la contabilità delle risorse.
"""

from collections.abc import Iterator
from .container import DockerContainer, ContainerState
from .function import Function


class ContainerPool:
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

    def _reusable_busy(self) -> DockerContainer | None:
        return next((c for c in self._busy
                     if c.requests_count < self.function.max_concurrency), None)

    def _add(self, container: DockerContainer) -> None:
        if container.function != self.function or container in self:
            raise ValueError('Container incompatibile o già presente')
        if container.state is ContainerState.BUSY:
            self._busy.append(container)
        elif container.state is ContainerState.IDLE:
            self._idle.append(container)
        else:
            raise ValueError('Container non pronto')

    def _pop_idle(self) -> DockerContainer:
        return self._idle.pop()

    def _remove_busy(self, container: DockerContainer) -> None:
        self._swap_remove(self._busy, container)

    def _remove_idle(self, container: DockerContainer) -> None:
        self._swap_remove(self._idle, container)

    @staticmethod
    def _swap_remove(items: list[DockerContainer], container: DockerContainer) -> None:
        index = items.index(container)
        items[index] = items[-1]
        items.pop()

    def _remove_expired(self, now: float) -> tuple[DockerContainer, ...]:
        expired = tuple(c for c in self._idle if c.is_expired(now))
        self._idle[:] = [c for c in self._idle if not c.is_expired(now)]
        return expired

    def __len__(self) -> int:
        return len(self._busy) + len(self._idle)

    def __iter__(self) -> Iterator[DockerContainer]:
        return iter(self.busy + self.idle)

    def __contains__(self, container: object) -> bool:
        return container in self._busy or container in self._idle
