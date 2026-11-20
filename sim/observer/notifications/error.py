"""Notifiche di errore durante l'esecuzione della simulazione."""

from dataclasses import dataclass, field
from logging import ERROR

from .notification import Notification, NotificationType


@dataclass(frozen=True)
class SimulationError(Notification):
    """Descrive un errore del calendario con dati serializzabili."""

    type = NotificationType.SIMULATION_ERROR
    level = ERROR

    exception: Exception = field(repr=False)
    failed_event: str
    run_id: str | None = None
    sim_time_s: float | None = None

    @property
    def error_type(self) -> str:
        return type(self.exception).__name__

    @property
    def message(self) -> str:
        return str(self.exception)

    def __str__(self) -> str:
        time = f'[{self.sim_time_s:.6g} s] ' if self.sim_time_s is not None else ''
        run = f'run={self.run_id} ' if self.run_id is not None else ''
        return f'{time}{self.type.value} {run}evento={self.failed_event} {self.error_type}: {self.message}'

    def to_dict(self) -> dict:
        return {
            'type': self.type.value,
            'run_id': self.run_id,
            'sim_time_s': self.sim_time_s,
            'data': {
                'failed_event': self.failed_event,
                'error_type': self.error_type,
                'message': self.message,
            },
        }
