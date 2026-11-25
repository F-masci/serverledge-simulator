"""Iperparametri dell'agente PPO (Ray RLlib).

Esempio TOML nello scenario (tutti i parametri sono opzionali):

    [regions.edge.scaler.ppo]
    hidden_sizes = [64, 64]                # opzionale (default: [64, 64])
    learning_rate = 0.0003                 # opzionale (default: 3e-4)
    gamma = 0.99                           # opzionale (default: 0.99, in [0, 1])
    gae_lambda = 0.95                      # opzionale (default: 0.95, in [0, 1])
    clip_epsilon = 0.2                     # opzionale (default: 0.2)
    value_coef = 0.5                       # opzionale (default: 0.5)
    entropy_coef = 0.01                    # opzionale (default: 0.01)
    max_grad_norm = 0.5                    # opzionale (default: 0.5)
    rollout_steps = 200                    # opzionale (default: 64, batch per iterazione RLlib)
    ppo_epochs = 4                         # opzionale (default: 4)
    minibatch_size = 50                    # opzionale (default: 16)
    normalize_advantages = true            # opzionale (default: true)
    training = true                        # opzionale (default: true)
    deterministic = false                  # opzionale (default: false; true in valutazione)
    checkpoint_path = "checkpoints/ppo_edge"  # opzionale (default: None, carica pesi addestrati)
"""

from collections.abc import Mapping
from dataclasses import dataclass

from sim.utils.validation import nonnegative_number, positive_integer, positive_number


@dataclass(frozen=True)
class PPOConfig:
    """Iperparametri dell'agente PPO (Ray RLlib)."""

    hidden_sizes: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 64
    ppo_epochs: int = 4
    minibatch_size: int = 16
    normalize_advantages: bool = True
    training: bool = True
    deterministic: bool = False
    checkpoint_path: str | None = None

    def __post_init__(self) -> None:
        if not self.hidden_sizes:
            raise ValueError('scaler.ppo.hidden_sizes deve contenere almeno un layer')
        for size in self.hidden_sizes:
            positive_integer(size, 'scaler.ppo.hidden_sizes')
        positive_number(self.learning_rate, 'scaler.ppo.learning_rate')
        if nonnegative_number(self.gamma, 'scaler.ppo.gamma') > 1.0:
            raise ValueError('scaler.ppo.gamma deve essere in [0, 1]')
        if nonnegative_number(self.gae_lambda, 'scaler.ppo.gae_lambda') > 1.0:
            raise ValueError('scaler.ppo.gae_lambda deve essere in [0, 1]')
        positive_number(self.clip_epsilon, 'scaler.ppo.clip_epsilon')
        nonnegative_number(self.value_coef, 'scaler.ppo.value_coef')
        nonnegative_number(self.entropy_coef, 'scaler.ppo.entropy_coef')
        positive_number(self.max_grad_norm, 'scaler.ppo.max_grad_norm')
        positive_integer(self.rollout_steps, 'scaler.ppo.rollout_steps')
        positive_integer(self.ppo_epochs, 'scaler.ppo.ppo_epochs')
        positive_integer(self.minibatch_size, 'scaler.ppo.minibatch_size')
        for field_name in ('normalize_advantages', 'training', 'deterministic'):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f'scaler.ppo.{field_name} deve essere un booleano')
        if self.checkpoint_path is not None and (
            not isinstance(self.checkpoint_path, str) or not self.checkpoint_path.strip()
        ):
            raise ValueError('scaler.ppo.checkpoint_path deve essere una stringa non vuota')

    @classmethod
    def parse(cls, data: object, *, prefix: str = 'scaler.ppo') -> 'PPOConfig':
        """Parse della configurazione degli iperparametri PPO."""
        if data is None:
            return cls()
        allowed = {
            'hidden_sizes', 'learning_rate', 'gamma', 'gae_lambda', 'clip_epsilon',
            'value_coef', 'entropy_coef', 'max_grad_norm', 'rollout_steps',
            'ppo_epochs', 'minibatch_size', 'normalize_advantages', 'training',
            'deterministic', 'checkpoint_path',
        }
        if not isinstance(data, Mapping) or data.keys() - allowed:
            raise ValueError(f'{prefix} ammette solo: {", ".join(sorted(allowed))}')
        raw_hidden = data.get('hidden_sizes', [64, 64])
        if not isinstance(raw_hidden, (list, tuple)) or not raw_hidden:
            raise ValueError(f'{prefix}.hidden_sizes deve essere una lista non vuota di interi positivi')
        hidden = tuple(positive_integer(h, f'{prefix}.hidden_sizes') for h in raw_hidden)
        raw_ckpt = data.get('checkpoint_path')
        return cls(
            hidden_sizes=hidden,
            learning_rate=positive_number(data.get('learning_rate', 3e-4), f'{prefix}.learning_rate'),
            gamma=nonnegative_number(data.get('gamma', 0.99), f'{prefix}.gamma'),
            gae_lambda=nonnegative_number(data.get('gae_lambda', 0.95), f'{prefix}.gae_lambda'),
            clip_epsilon=positive_number(data.get('clip_epsilon', 0.2), f'{prefix}.clip_epsilon'),
            value_coef=nonnegative_number(data.get('value_coef', 0.5), f'{prefix}.value_coef'),
            entropy_coef=nonnegative_number(data.get('entropy_coef', 0.01), f'{prefix}.entropy_coef'),
            max_grad_norm=positive_number(data.get('max_grad_norm', 0.5), f'{prefix}.max_grad_norm'),
            rollout_steps=positive_integer(data.get('rollout_steps', 64), f'{prefix}.rollout_steps'),
            ppo_epochs=positive_integer(data.get('ppo_epochs', 4), f'{prefix}.ppo_epochs'),
            minibatch_size=positive_integer(data.get('minibatch_size', 16), f'{prefix}.minibatch_size'),
            normalize_advantages=data.get('normalize_advantages', True),
            training=data.get('training', True),
            deterministic=data.get('deterministic', False),
            checkpoint_path=raw_ckpt if raw_ckpt is not None else None,
        )
