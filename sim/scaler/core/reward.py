"""Calcolo della reward a due termini della rev-1."""

from dataclasses import dataclass

from ..config import RewardConfig
from ..contracts import ZoneObservation


@dataclass(frozen=True)
class RewardBreakdown:
    """Scomposizione della reward a due termini per l'intervallo concluso."""

    k_cost: float
    d_cost_req_s: float
    reward: float


class RewardCalculator:
    """Calcola la reward a due termini: r_t = -[w_K * K_t / K_ref + w_T * D_t / D_ref]."""

    def __init__(self, config: RewardConfig) -> None:
        self._config = config

    @property
    def config(self) -> RewardConfig:
        return self._config

    def compute(self, observation: ZoneObservation) -> RewardBreakdown:
        """Calcola K_t, D_t e reward scalare da una ZoneObservation."""

        # Calcolo dei costi K_t e D_t
        k_cost = observation.interval_energy_kwh + observation.interval_allocation_cost
        d_cost = observation.interval_wait_req_s + observation.interval_rejection_penalty_req_s

        # Calcolo della reward scalare
        reward_config = self._config
        reward = -(reward_config.w_k * (k_cost / reward_config.k_ref) + reward_config.w_t * (d_cost / reward_config.d_ref))

        return RewardBreakdown(
            k_cost=k_cost,
            d_cost_req_s=d_cost,
            reward=reward,
        )
