"""Configurazione della funzione di reward a due termini della rev-1.

Esempio TOML nello scenario (tutti i parametri sono opzionali):

    [regions.edge.scaler.reward]
    w_k = 1.0                # opzionale (default: 1.0, peso costo energetico K_t)
    w_t = 1.0                # opzionale (default: 1.0, peso costo temporale D_t)
    k_ref = 0.01             # opzionale (default: 1.0, scala positiva di riferimento per K_t in kWh)
    d_ref = 10.0             # opzionale (default: 1.0, scala positiva di riferimento per D_t in req*s)
    timeout_penalty_s = 10.0 # opzionale (default: 10.0 s, T_timeout equivalente per rifiuti)
"""

from collections.abc import Mapping
from dataclasses import dataclass

from simulator.utils.validation import nonnegative_number, positive_number


@dataclass(frozen=True)
class RewardConfig:
    """Parametri della reward a due termini: r_t = -[w_K * K_t / K_ref + w_T * D_t / D_ref]."""

    w_k: float = 1.0
    w_t: float = 1.0
    k_ref: float = 1.0
    d_ref: float = 1.0
    timeout_penalty_s: float = 10.0

    def __post_init__(self) -> None:
        nonnegative_number(self.w_k, 'scaler.reward.w_k')
        nonnegative_number(self.w_t, 'scaler.reward.w_t')
        positive_number(self.k_ref, 'scaler.reward.k_ref')
        positive_number(self.d_ref, 'scaler.reward.d_ref')
        nonnegative_number(self.timeout_penalty_s, 'scaler.reward.timeout_penalty_s')

    @classmethod
    def parse(cls, data: object, *, prefix: str = 'scaler.reward') -> 'RewardConfig':
        """Parse della configurazione della funzione di reward."""
        if data is None:
            return cls()
        allowed = {'w_k', 'w_t', 'k_ref', 'd_ref', 'timeout_penalty_s'}
        if not isinstance(data, Mapping) or data.keys() - allowed:
            raise ValueError(f'{prefix} ammette solo: {", ".join(sorted(allowed))}')
        return cls(
            w_k=nonnegative_number(data.get('w_k', 1.0), f'{prefix}.w_k'),
            w_t=nonnegative_number(data.get('w_t', 1.0), f'{prefix}.w_t'),
            k_ref=positive_number(data.get('k_ref', 1.0), f'{prefix}.k_ref'),
            d_ref=positive_number(data.get('d_ref', 1.0), f'{prefix}.d_ref'),
            timeout_penalty_s=nonnegative_number(
                data.get('timeout_penalty_s', 10.0), f'{prefix}.timeout_penalty_s',
            ),
        )
