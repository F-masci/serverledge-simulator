"""Configurazione complessiva dello scaler regionale.

Esempio TOML nello scenario (globale `[scaler]` o per regione `[regions.<nome>.scaler]`):

    [regions.edge.scaler]
    mode = "PPO"             # opzionale (default: "None" se omesso, "PPO" nella tabella | "DDQN" | "Rainbow")
    interval_s = 5.0         # opzionale (default: 5.0 s, intervallo Delta)
    history_length = 1       # opzionale (default: 1, passi impilati nell'osservazione)

    # Sottosezioni opzionali:
    # [regions.edge.scaler.reward]       (vedi scaler/config/reward.py)
    # [regions.edge.scaler.ppo]          (vedi scaler/config/ppo.py)
    # [regions.edge.scaler.default_ring] (vedi simulator/io/scaler/ring.py)
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from scaler.config import PPOConfig, RewardConfig
from simulator.utils.validation import positive_integer, positive_number

from .ring import RingScalingConfig


class ScalerMode(StrEnum):
    """Algoritmi di autoscaling selezionabili per una regione."""
    NONE = 'None'        # Autoscaling disabilitato (topologia statica).
    PPO = 'PPO'          # Proximal Policy Optimization (Ray RLlib).
    DDQN = 'DDQN'        # Double DQN (non ancora implementato).
    RAINBOW = 'Rainbow'  # Rainbow DQN (non ancora implementato).


@dataclass(frozen=True)
class ScalerConfig:
    """Configurazione completa dello scaler di una regione."""

    mode: ScalerMode = ScalerMode.NONE
    interval_s: float = 5.0
    history_length: int = 1
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    default_ring: RingScalingConfig = field(default_factory=RingScalingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ScalerMode):
            raise ValueError(f'scaler.mode deve essere uno tra {", ".join(ScalerMode)}')
        positive_number(self.interval_s, 'scaler.interval_s')
        positive_integer(self.history_length, 'scaler.history_length')

    @property
    def is_enabled(self) -> bool:
        """Indica se lo scaler è attivo per la regione."""
        return self.mode is not ScalerMode.NONE

    @classmethod
    def parse(cls, data: object, *, prefix: str = 'scaler') -> 'ScalerConfig':
        """Parse della sezione scaler dello scenario o di una regione."""
        if data is None:
            return cls()
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            try:
                mode = ScalerMode(data)
            except ValueError as error:
                modes = ', '.join(ScalerMode)
                raise ValueError(f'{prefix} deve essere uno tra {modes}') from error
            return cls(mode=mode)

        allowed = {'mode', 'interval_s', 'history_length', 'reward', 'ppo', 'default_ring'}
        if not isinstance(data, Mapping) or data.keys() - allowed:
            raise ValueError(f'{prefix} ammette solo: {", ".join(sorted(allowed))}')
        raw_mode = data.get('mode', ScalerMode.PPO.value)
        try:
            mode = ScalerMode(raw_mode)
        except ValueError as error:
            modes = ', '.join(ScalerMode)
            raise ValueError(f'{prefix}.mode deve essere uno tra {modes}') from error
        return cls(
            mode=mode,
            interval_s=positive_number(data.get('interval_s', 5.0), f'{prefix}.interval_s'),
            history_length=positive_integer(data.get('history_length', 1), f'{prefix}.history_length'),
            reward=RewardConfig.parse(data.get('reward'), prefix=f'{prefix}.reward'),
            ppo=PPOConfig.parse(data.get('ppo'), prefix=f'{prefix}.ppo'),
            default_ring=RingScalingConfig.parse(data.get('default_ring'), prefix=f'{prefix}.default_ring'),
        )
