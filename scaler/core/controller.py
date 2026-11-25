"""Core decisionale dello Zone Controller, indipendente dal simulatore."""

import numpy as np

from .codec import ActionCodec
from .encoder import ObservationEncoder
from .reward import RewardBreakdown, RewardCalculator
from ..config import RewardConfig
from ..contracts import (
    ScalingBackend, ScalingCommand, ScalingPolicy, ScalingResult, ZoneObservation,
)


class ZoneController:
    """Core decisionale dello Zone Controller."""

    @property
    def codec(self) -> ActionCodec:
        return self._codec

    def __init__(
        self,
        *,
        region_id: str,
        ring_ids: tuple[str, ...],
        policy: ScalingPolicy,
        reward_config: RewardConfig,
        interval_s: float,
        history_length: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        # Inizializza l'ID della regione.
        self._region_id = region_id

        # Inizializza la policy di scaling.
        self._policy = policy
        self._codec = ActionCodec(ring_ids)
        self._encoder = ObservationEncoder(ring_ids, reward_config, interval_s=interval_s, history_length=history_length)
        self._reward_calc = RewardCalculator(reward_config)

        # Inizializza il generatore di numeri casuali.
        self._rng = np.random.default_rng() if rng is None else rng

        # Inizializza l'ID della decisione.
        self._decision_id = 0

    @property
    def encoder(self) -> ObservationEncoder:
        return self._encoder

    @property
    def reward_calculator(self) -> RewardCalculator:
        return self._reward_calc

    @property
    def policy(self) -> ScalingPolicy:
        return self._policy

    @property
    def decisions_count(self) -> int:
        return self._decision_id

    def decide(
        self,
        observation: ZoneObservation,
        *,
        record_history: bool = True,
        deterministic: bool | None = None,
    ) -> tuple[ScalingCommand, RewardBreakdown]:
        """Calcola reward, codifica l'osservazione mascherata e genera il comando di scaling."""

        # Calcola il reward e il breakdown associato all'osservazione corrente.
        breakdown = self._reward_calc.compute(observation)

        # Codifica l'osservazione.
        obs_vec = self._encoder.push(observation) if record_history else self._encoder.peek(observation)

        # Calcola la maschera delle azioni disponibili.
        mask = self._codec.action_mask(observation)

        # Seleziona l'azione da eseguire tramite la policy.
        action_idx = self._policy.select_action(obs_vec, mask, rng=self._rng, deterministic=deterministic)

        # Decodifica l'azione selezionata in un comando di scaling.
        kind, ring_id = self._codec.decode(action_idx)

        # Incrementa l'ID della decisione.
        self._decision_id += 1

        # Crea il comando di scaling da restituire.
        command = ScalingCommand(
            command_id=f'{self._region_id}-cmd-{self._decision_id}',
            decision_id=self._decision_id,
            ring_id=ring_id,
            action=kind,
            state_version=observation.state_version,
            issued_at_s=observation.timestamp_s,
        )

        return command, breakdown

    def control_step(
        self,
        backend: ScalingBackend,
        *,
        deterministic: bool | None = None,
    ) -> tuple[ZoneObservation, ScalingCommand, ScalingResult, RewardBreakdown]:
        """Esegue un intero ciclo periodico su qualunque `ScalingBackend` (simulatore o gRPC)."""
        observation = backend.observe()
        command, breakdown = self.decide(observation, record_history=True, deterministic=deterministic)
        result = backend.execute(command)
        return observation, command, result, breakdown
