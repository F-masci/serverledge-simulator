"""DTO immutabili delle osservazioni di ring e di zona (mappabili 1-a-1 su Protobuf/gRPC)."""

from dataclasses import dataclass

FEATURES_PER_RING = 13


@dataclass(frozen=True)
class RingObservation:
    """Fotografia aggregata di un singolo ring di nodi nell'ultimo intervallo Delta."""

    ring_id: str
    recent_arrivals: int
    backlog_count: int
    backlog_max_age_s: float
    cpu_utilization: float
    memory_utilization: float
    ready_nodes: int
    provisioning_nodes: int
    draining_nodes: int
    est_remaining_boot_s: float
    remaining_quota: int
    min_nodes: int
    max_nodes: int
    boot_time_s: float
    interval_energy_kwh: float
    interval_delay_req_s: float
    interval_rejections: int
    can_scale_out: bool
    can_scale_in: bool

    def to_metrics_dict(self) -> dict[str, float | int]:
        """Esporta le metriche osservate del ring in formato serializzabile."""
        return {
            'recent_arrivals': self.recent_arrivals,
            'backlog_count': self.backlog_count,
            'backlog_max_age_s': self.backlog_max_age_s,
            'cpu_utilization': self.cpu_utilization,
            'memory_utilization': self.memory_utilization,
            'ready_nodes': self.ready_nodes,
            'provisioning_nodes': self.provisioning_nodes,
            'draining_nodes': self.draining_nodes,
            'est_remaining_boot_s': self.est_remaining_boot_s,
            'remaining_quota': self.remaining_quota,
            'interval_energy_kwh': self.interval_energy_kwh,
            'interval_delay_req_s': self.interval_delay_req_s,
            'interval_rejections': self.interval_rejections,
        }


@dataclass(frozen=True)
class ZoneObservation:
    """Osservazione aggregata di una regione/zona all'istante decisionale."""

    region_id: str
    timestamp_s: float
    state_version: int
    rings: tuple[RingObservation, ...]
    interval_energy_kwh: float = 0.0
    interval_allocation_cost: float = 0.0
    interval_wait_req_s: float = 0.0
    interval_rejection_penalty_req_s: float = 0.0
