"""Interfacce astratte condivise tra simulatore e sistema reale (gRPC)."""

from abc import ABC, abstractmethod

import numpy as np

from .actions import ScalingCommand, ScalingResult
from .observations import ZoneObservation


class ScalingPolicy(ABC):
    """Contratto comune per tutte le policy decisionali (PPO, DDQN, Rainbow, soglie, ecc.)."""

    @abstractmethod
    def select_action(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        *,
        rng: np.random.Generator,
        deterministic: bool | None = None,
    ) -> int:
        """Seleziona un indice di azione ammissibile dato osservazione e maschera."""


class ScalingBackend(ABC):
    """Contratto comune verso l'attuazione: simulatore oppure Infra Controller Go via gRPC."""

    @property
    @abstractmethod
    def region_id(self) -> str:
        """Identificativo della regione/zona gestita."""

    @property
    @abstractmethod
    def ring_ids(self) -> tuple[str, ...]:
        """Elenco ordinato e deterministico degli ID dei ring della zona."""

    @abstractmethod
    def observe(self) -> ZoneObservation:
        """Raccoglie l'osservazione aggregata corrente dai ring della zona."""

    @abstractmethod
    def execute(self, command: ScalingCommand) -> ScalingResult:
        """Invia il comando all'Infra Controller del ring destinatario e restituisce l'esito."""
