"""Stato logico di un container Docker, senza accesso al daemon."""
from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from typing import TYPE_CHECKING

from simulator.utils.validation import finite_time

from .function import Function

if TYPE_CHECKING:
    from ..topology.node import Node
    from ..topology.ring import ArchitectureRing
    from ..topology.infrastructure import Infrastructure
    from ..topology.region import Region


class ContainerState(Enum):
    """Stato logico di un container Docker."""
    CREATING = 'creating'    # Risorse prenotate, creazione del container ancora in corso.
    BUSY = 'busy'            # Container pronto con almeno un'invocazione attiva.
    IDLE = 'idle'            # Nessuna invocazione attiva, disponibile nel pool warm fino alla rimozione.
    DESTROYED = 'destroyed'  # Container rimosso oppure creazione fallita.


class DockerContainer:
    """
    Rappresenta lo stato di un container Docker
    """

    def __init__(self, container_id: str, function: Function, *, node: 'Node | None' = None) -> None:
        self._id = container_id
        self._node = node
        self._function = function
        self._state = ContainerState.CREATING
        self._requests_count = 0
        self._expiration_time: float | None = None
        self._evicted = False

        ### Callback ###
        
        self.on_created: Callable[[DockerContainer], None] | None = None
        self.on_busy: Callable[[DockerContainer], None] | None = None
        self.on_idle: Callable[[DockerContainer], None] | None = None
        self.on_destroyed: Callable[[DockerContainer], None] | None = None
        self.on_state_changed: Callable[[DockerContainer, ContainerState, ContainerState], None] | None = None

        self.on_acquire: Callable[[DockerContainer], None] | None = None

    ### Identification ###

    @property
    def id(self) -> str:
        return self._id

    ### Infrastructure ###

    @property
    def node(self) -> Node:
        assert self._node is not None, 'Container non associato a un nodo'
        return self._node

    @property
    def ring(self) -> ArchitectureRing:
        return self.node.ring

    @property
    def infrastructure(self) -> Infrastructure:
        return self.ring.infrastructure

    @property
    def region(self) -> Region:
        return self.infrastructure.region

    ### Container State ###

    @property
    def state(self) -> ContainerState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state is ContainerState.IDLE

    @property
    def is_busy(self) -> bool:
        return self._state is ContainerState.BUSY

    @property
    def is_warm(self) -> bool:
        return self._state in (ContainerState.IDLE, ContainerState.BUSY)

    def is_expired(self, now: float) -> bool:
        """Verifica se il container è scaduto."""
        now = finite_time(now, 'now')
        expiration_time = self.expiration_time

        # Un container è considerato scaduto se è in stato IDLE
        # e il tempo simulato corrente supera il tempo di scadenza.
        if self._state is not ContainerState.IDLE:
            return False
        if expiration_time is None:
            return False
        return now > expiration_time

    ### Execution State ###

    @property
    def function(self) -> Function:
        return self._function

    @property
    def requests_count(self) -> int:
        return self._requests_count

    @property
    def expiration_time(self) -> float | None:
        """Tempo simulato in secondi"""
        return self._expiration_time

    @property
    def evicted(self) -> bool:
        """Indica se il container è stato distrutto per pressione di memoria (eviction)."""
        return self._evicted

    @property
    def memory_limit_bytes(self) -> int:
        return self.function.memory_mb * 1048576

    @property
    def cpu_period_us(self) -> int:
        return 50000

    @property
    def cpu_quota_us(self) -> int:
        return int(self.cpu_period_us * self.function.cpu_demand)

    #################################
    ### Internal state management ###
    #################################

    ## State transitions ##

    def created(self) -> None:
        """Segna il container come creato e pronto per la prima esecuzione."""
        self._ensure_status(ContainerState.CREATING, 'Creazione container')

        # Imposta il contatore prima di notificare la creazione e lo stato operativo.
        self._requests_count = 1
        self._expiration_time = None
        self._handle_change_status(ContainerState.BUSY, self.on_created)

        if self.on_busy is not None:
            self.on_busy(self)

    def acquire(self) -> None:
        """Segna il container come occupato da una richiesta."""
        self._ensure_statuses((ContainerState.IDLE, ContainerState.BUSY), 'Container non riutilizzabile')
        self._ensure_concurrency()
        
        # Ogni acquisizione notifica on_busy, anche con altre richieste attive.
        self._requests_count += 1
        self._expiration_time = None
        if self.on_acquire is not None:
            self.on_acquire(self)
        self._handle_change_status(ContainerState.BUSY, self.on_busy)

    def complete(self, now: float, timeout: float) -> None:
        """Segna il completamento di una richiesta e aggiorna lo stato del container."""
        self._ensure_status(ContainerState.BUSY, 'Nessuna richiesta attiva')

        # Il container torna idle alla conclusione dell'ultima richiesta attiva.
        self._requests_count -= 1
        if self.requests_count == 0:
            self._expiration_time = now + timeout
            self._handle_change_status(ContainerState.IDLE, self.on_idle)

    def destroy(self, *, evicted: bool = False) -> None:
        """Segnala il container come distrutto."""
        self._ensure_statuses((ContainerState.IDLE, ContainerState.CREATING), 'Distruzione ordinaria consentita solo per idle o creazione fallita')

        # Registra la causa della distruzione e notifica prima di scollegare il nodo.
        self._evicted = evicted
        self._handle_change_status(ContainerState.DESTROYED, self.on_destroyed)
        self._node = None

    ## Callback handling ##

    def _handle_change_status(
        self, state: ContainerState, callback: Callable[[DockerContainer], None] | None,
    ) -> None:
        """Aggiorna lo stato e invoca le callback nell'ordine previsto."""
        previous_state = self._state
        self._state = state

        # La callback specifica precede la notifica del cambio di stato.
        if callback is not None:
            callback(self)
        if self.on_state_changed is not None and previous_state is not state:
            self.on_state_changed(self, previous_state, state)

    ######################
    ### Error handling ###
    ######################

    def _ensure_status(self, expected: ContainerState, message: str) -> None:
        if self._state is not expected:
            raise ValueError(f'{message}: stato attuale={self._state.value}, atteso={expected.value}')

    def _ensure_statuses(self, expected: tuple[ContainerState, ...], message: str) -> None:
        if self._state not in expected:
            expected_values = ', '.join(state.value for state in expected)
            raise ValueError(f'{message}: stato attuale={self._state.value}, attesi={expected_values}')

    def _ensure_concurrency(self) -> None:
        if self._requests_count >= self.function.max_concurrency:
            raise ValueError('Concorrenza massima raggiunta')

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile del container."""
        fields = self.to_dict()
        details = ', '.join(f'{name}={value}' for name, value in fields.items() if value is not None)
        return f'DockerContainer({details})'

    def to_dict(self) -> dict:
        """Restituisce lo stato corrente del container in un formato serializzabile."""
        return {
            'id': self.id,
            'function': self.function.name,
            'state': self.state.value,
            'requests_count': self.requests_count,
            'expiration_time_s': self.expiration_time,
        }

    def to_json(self) -> str:
        """Serializza il container per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<DockerContainer id={self.id} function={self.function.name} state={self.state.value}>'
