"""Notifiche che trasportano le metriche di una replica."""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Literal

from .notification import Notification, NotificationType


@dataclass(frozen=True)
class FinalStatistics(Notification):
    """Fotografia delle metriche al termine dell'esecuzione."""

    type = NotificationType.FINAL_STATISTICS

    run_id: str
    sim_time_s: float
    outcome: Literal['completed', 'interrupted', 'failed']
    metrics: dict = field(repr=False)
    summary: str = field(repr=False)

    def __str__(self) -> str:
        return f'[{self.sim_time_s:.6g} s] {self.type.value} run={self.run_id} esito={self.outcome}\n{self.summary}'

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {'outcome': self.outcome, 'metrics': deepcopy(self.metrics)},
        }
