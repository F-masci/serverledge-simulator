"""Notifiche emesse dallo scaler regionale ad ogni passo decisionale."""

from dataclasses import dataclass
from logging import INFO
from typing import TYPE_CHECKING

from .notification import Notification, NotificationType

if TYPE_CHECKING:
    from sim.model.region import Region


@dataclass(frozen=True)
class ScalerDecision(Notification):
    """Notifica una decisione dello Zone Controller regionale con reward e costi dell'intervallo."""

    type = NotificationType.SCALER_DECISION
    level = INFO

    run_id: str
    sim_time_s: float
    region: 'Region'
    decision_id: int
    algorithm: str
    action: str
    ring_id: str | None
    outcome: str
    node_name: str | None
    reward: float | None
    k_cost: float | None
    d_cost_req_s: float | None
    energy_kwh: float | None

    def __str__(self) -> str:
        target = f' ring={self.ring_id}' if self.ring_id is not None else ''
        node = f' nodo={self.node_name}' if self.node_name is not None else ''
        rew = f' reward={self.reward:.4g}' if self.reward is not None else ''
        return (
            f'[{self.sim_time_s:.6g} s] {self.type.value} '
            f'regione={self.region.name} #{self.decision_id} {self.action}{target}{node} '
            f'esito={self.outcome}{rew}'
        )

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {
                'region': self.region.name,
                'decision_id': self.decision_id,
                'algorithm': self.algorithm,
                'action': self.action,
                'ring_id': self.ring_id,
                'outcome': self.outcome,
                'node_name': self.node_name,
                'reward': self.reward,
                'k_cost': self.k_cost,
                'd_cost_req_s': self.d_cost_req_s,
                'energy_kwh': self.energy_kwh,
            },
        }
