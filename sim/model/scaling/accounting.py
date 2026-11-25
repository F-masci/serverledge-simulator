"""Contabilità continua di energia K_t, attese D_t, reward e osservazioni per regione."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.observer.notifications import (
    ContainerStateChanged, NodeRegistered, NodeRemoved, Notification, RequestAccepted,
    RequestArrived, RequestCompleted, RequestDelivered, RequestForwarded, RequestQueued,
    RequestRejected,
)
from .actuator import RegionActuator
from sim.scaler.config import RewardConfig
from sim.scaler.contracts import FEATURES_PER_RING, RingObservation, ZoneObservation
from sim.scaler.core import ObservationEncoder, RewardCalculator

_JOULES_PER_KWH = 3_600_000.0


@dataclass
class _NodePowerState:
    """Stato di potenza istantanea (W) e costo di allocazione di un nodo nel tempo."""

    ring_id: str
    used_cpus: float
    total_cpus: float
    draining: bool = False


@dataclass
class IntervalSnapshot:
    """Riepilogo chiuso di un intervallo di controllo [t - Delta, t]."""

    start_s: float
    end_s: float
    energy_kwh: float
    allocation_cost: float
    k_cost: float
    wait_req_s: float
    rejection_penalty_req_s: float
    d_cost_req_s: float
    reward: float
    ring_observations: dict[str, dict[str, float | int]]
    feature_vector: np.ndarray


class RegionAccounting:
    """
    Traccia in tempo continuo costi infrastrutturali K_t, tempi di permanenza D_t
    e osservazioni per ring di una singola regione.

    Implementa la reward a due termini della rev-1:
        r_t = -[w_K * K_t / K_ref + w_T * D_t / D_ref]
    dove:
      - K_t integra energia (kWh) su tutte le fasi (idle, esecuzione, boot, drain)
        più l'eventuale costo di allocazione;
      - D_t integra N_pendenti(u) du (richiesta-secondi) più max(0, T_timeout - a)
        per ogni richiesta rifiutata/fallita di età a.
    """

    def __init__(
        self,
        actuator: RegionActuator,
        reward_config: RewardConfig,
        *,
        interval_s: float,
        history_length: int = 1,
    ) -> None:
        self._actuator = actuator
        self._region = actuator.region
        self._reward_config = reward_config
        self._interval_s = interval_s
        self._history_length = history_length

        # Istante di avanzamento degli integrali continui e inizio intervallo corrente.
        self._last_time_s = 0.0
        self._interval_start_s = 0.0

        # Stato potenza nodi attivi e in drain (chiave = node.key).
        self._node_power: dict[str, _NodePowerState] = {}
        infra = self._region.infrastructure
        assert infra is not None
        for ring_id, ring in infra.rings.items():
            for node in ring:
                self._node_power[node.key] = _NodePowerState(
                    ring_id=ring_id,
                    used_cpus=node.resources.used_cpus,
                    total_cpus=node.resources.total_cpus,
                    draining=False,
                )

        # Richieste pendenti nella regione: request_id -> (enter_time_s, ring_id | None).
        self._pending: dict[str, tuple[float, str | None]] = {}

        # Accumulatori di intervallo per ring e totali regionali.
        self._interval_energy_j_by_ring: dict[str, float] = {r: 0.0 for r in actuator.ring_order}
        self._interval_alloc_by_ring: dict[str, float] = {r: 0.0 for r in actuator.ring_order}
        self._interval_wait_req_s_by_ring: dict[str, float] = {r: 0.0 for r in actuator.ring_order}
        self._interval_rej_pen_by_ring: dict[str, float] = {r: 0.0 for r in actuator.ring_order}
        self._interval_arrivals_by_ring: dict[str, int] = {r: 0 for r in actuator.ring_order}
        self._interval_rejections_by_ring: dict[str, int] = {r: 0 for r in actuator.ring_order}
        self._interval_wait_req_s_total = 0.0
        self._interval_rej_pen_total = 0.0

        # Accumulatori cumulativi dell'intera replica.
        self.total_energy_kwh = 0.0
        self.total_allocation_cost = 0.0
        self.total_k_cost = 0.0
        self.total_wait_req_s = 0.0
        self.total_rejection_penalty_req_s = 0.0
        self.total_d_cost_req_s = 0.0
        self.total_reward = 0.0
        self.intervals_closed = 0
        self.total_energy_kwh_by_ring: dict[str, float] = {r: 0.0 for r in actuator.ring_order}
        self.total_d_cost_by_ring: dict[str, float] = {r: 0.0 for r in actuator.ring_order}

        # Encoder e calcolatore reward condivisi con il core indipendente dal simulatore.
        self._encoder = ObservationEncoder(
            actuator.ring_order, reward_config, interval_s=interval_s, history_length=history_length,
        )
        self._reward_calc = RewardCalculator(reward_config)

    ### Properties ###

    @property
    def obs_dim(self) -> int:
        """Dimensione del vettore di osservazione concatenato sullo storico breve."""
        return self._encoder.obs_dim

    ### Continuous-Time Integration ###

    def _instant_power_w(self, state: _NodePowerState) -> float:
        """Calcola la potenza istantanea (W) del nodo in base a fase e utilizzo CPU."""
        cfg = self._actuator.rings[state.ring_id].config
        base_w = cfg.effective_drain_power_w if state.draining else cfg.idle_power_w
        util = min(1.0, max(0.0, state.used_cpus / state.total_cpus)) if state.total_cpus > 0 else 0.0
        return base_w + max(0.0, cfg.max_power_w - base_w) * util

    def advance_to(self, now: float) -> None:
        """Avanza gli integrali continui di energia, allocazione e richieste pendenti fino a now."""
        if now <= self._last_time_s:
            return
        dt = now - self._last_time_s
        self._last_time_s = now

        # 1. Energia e costo di allocazione dei nodi pronti e in drain.
        for state in self._node_power.values():
            ring_id = state.ring_id
            self._interval_energy_j_by_ring[ring_id] += self._instant_power_w(state) * dt
            alloc_rate = self._actuator.rings[ring_id].config.allocation_cost_per_s
            if alloc_rate > 0.0:
                self._interval_alloc_by_ring[ring_id] += alloc_rate * dt

        # 2. Energia e costo di allocazione dei nodi in acquisizione/boot.
        for ring_id, ring_act in self._actuator.rings.items():
            prov_count = ring_act.provisioning_count
            if prov_count > 0:
                self._interval_energy_j_by_ring[ring_id] += (
                    prov_count * ring_act.config.effective_boot_power_w * dt
                )
                if ring_act.config.allocation_cost_per_s > 0.0:
                    self._interval_alloc_by_ring[ring_id] += (
                        prov_count * ring_act.config.allocation_cost_per_s * dt
                    )

        # 3. Integrale continuo delle richieste pendenti N_pendenti(u) du.
        n_pending = len(self._pending)
        if n_pending > 0:
            self._interval_wait_req_s_total += n_pending * dt
            for _, ring_id in self._pending.values():
                if ring_id is not None and ring_id in self._interval_wait_req_s_by_ring:
                    self._interval_wait_req_s_by_ring[ring_id] += dt

    ### Actuator Hooks ###

    def on_node_drain_completed(self, node_key: str, now: float) -> None:
        """Rimuove un nodo drenato dalla contabilità energetica dopo aver chiuso l'intervallo."""
        self.advance_to(now)
        self._node_power.pop(node_key, None)

    ### Observer Notification Handling ###

    def on_notification(self, notification: Notification) -> None:
        """Aggiorna in tempo continuo contabilità energetica, richieste pendenti e drain."""
        now = notification.sim_time_s
        region_name = self._region.name

        match notification:
            case NodeRegistered():
                if notification.region.name != region_name:
                    return
                self.advance_to(now)
                self._node_power[notification.node.key] = _NodePowerState(
                    ring_id=notification.ring.id,
                    used_cpus=notification.resources.used_cpus,
                    total_cpus=notification.resources.total_cpus,
                    draining=False,
                )

            case NodeRemoved():
                if notification.region.name != region_name:
                    return
                self.advance_to(now)
                state = self._node_power.get(notification.node.key)
                if state is not None:
                    state.draining = True
                    state.used_cpus = notification.resources.used_cpus

            case ContainerStateChanged():
                if notification.region is None or notification.region.name != region_name:
                    return
                self.advance_to(now)
                state = self._node_power.get(notification.node.key)
                if state is not None and notification.resources is not None:
                    state.used_cpus = notification.resources.used_cpus
                self._actuator.check_node_drain(notification.node, now)

            case RequestArrived():
                if notification.region is None or notification.region.name != region_name:
                    return
                self.advance_to(now)
                self._pending[notification.request.id] = (now, None)

            case RequestDelivered():
                if notification.region is None or notification.region.name != region_name:
                    return
                self.advance_to(now)
                self._pending.setdefault(notification.request.id, (now, None))

            case RequestForwarded():
                if notification.source_region.name != region_name:
                    return
                self.advance_to(now)
                self._pending.pop(notification.request.id, None)

            case RequestQueued():
                if notification.region is None or notification.region.name != region_name:
                    return
                self.advance_to(now)
                ring_id = notification.ring.id if notification.ring is not None else None
                enter_t, prev_ring = self._pending.get(notification.request.id, (now, None))
                self._pending[notification.request.id] = (enter_t, ring_id)
                if prev_ring is None and ring_id in self._interval_arrivals_by_ring:
                    self._interval_arrivals_by_ring[ring_id] += 1

            case RequestAccepted():
                if notification.region is None or notification.region.name != region_name:
                    return
                self.advance_to(now)
                ring_id = notification.ring.id if notification.ring is not None else None
                enter_t, prev_ring = self._pending.get(notification.request.id, (now, None))
                self._pending[notification.request.id] = (enter_t, ring_id)
                if prev_ring is None and ring_id in self._interval_arrivals_by_ring:
                    self._interval_arrivals_by_ring[ring_id] += 1

            case RequestCompleted():
                if notification.request.id in self._pending:
                    self.advance_to(now)
                    self._pending.pop(notification.request.id, None)
                if (
                    notification.node is not None
                    and notification.region is not None
                    and notification.region.name == region_name
                ):
                    self._actuator.check_node_drain(notification.node, now)

            case RequestRejected():
                if notification.request.id not in self._pending and (
                    notification.region is None or notification.region.name != region_name
                ):
                    return
                self.advance_to(now)
                entry = self._pending.pop(notification.request.id, None)
                enter_t = entry[0] if entry is not None else now
                ring_id = (
                    notification.ring.id
                    if notification.ring is not None
                    else (entry[1] if entry is not None else None)
                )
                age = max(0.0, now - enter_t)
                penalty = max(0.0, self._reward_config.timeout_penalty_s - age)
                self._interval_rej_pen_total += penalty
                if ring_id is not None and ring_id in self._interval_rej_pen_by_ring:
                    self._interval_rej_pen_by_ring[ring_id] += penalty
                    self._interval_rejections_by_ring[ring_id] += 1

    ### Interval Closure & Observation Construction ###

    def zone_observation(self, now: float) -> ZoneObservation:
        """Costruisce il DTO immutabile `ZoneObservation` (compatibile con gRPC/infrascaler)."""
        self.advance_to(now)
        energy_kwh_by_ring = {
            r: joules / _JOULES_PER_KWH for r, joules in self._interval_energy_j_by_ring.items()
        }
        backlog_by_ring: dict[str, list[float]] = {r: [] for r in self._actuator.ring_order}
        for enter_t, ring_id in self._pending.values():
            if ring_id is not None and ring_id in backlog_by_ring:
                backlog_by_ring[ring_id].append(max(0.0, now - enter_t))

        rings: list[RingObservation] = []
        for ring_id in self._actuator.ring_order:
            ring_act = self._actuator.rings[ring_id]
            ready_nodes = tuple(ring_act.ring)
            total_cpus = sum(n.resources.total_cpus for n in ready_nodes)
            used_cpus = sum(n.resources.used_cpus for n in ready_nodes)
            total_mem = sum(n.resources.total_memory for n in ready_nodes)
            busy_mem = sum(n.resources.busy_pool_used_mem for n in ready_nodes)
            ages = backlog_by_ring[ring_id]
            rings.append(RingObservation(
                ring_id=ring_id,
                recent_arrivals=self._interval_arrivals_by_ring[ring_id],
                backlog_count=len(ages),
                backlog_max_age_s=max(ages) if ages else 0.0,
                cpu_utilization=used_cpus / total_cpus if total_cpus > 0 else 0.0,
                memory_utilization=busy_mem / total_mem if total_mem > 0 else 0.0,
                ready_nodes=ring_act.ready_count,
                provisioning_nodes=ring_act.provisioning_count,
                draining_nodes=ring_act.draining_count,
                est_remaining_boot_s=ring_act.estimated_remaining_boot_s(now),
                remaining_quota=ring_act.remaining_quota,
                min_nodes=ring_act.min_nodes,
                max_nodes=ring_act.max_nodes,
                boot_time_s=ring_act.config.boot_time_s,
                interval_energy_kwh=energy_kwh_by_ring[ring_id],
                interval_delay_req_s=(
                    self._interval_wait_req_s_by_ring[ring_id] + self._interval_rej_pen_by_ring[ring_id]
                ),
                interval_rejections=self._interval_rejections_by_ring[ring_id],
                can_scale_out=ring_act.can_scale_out(now),
                can_scale_in=ring_act.can_scale_in(now),
            ))

        return ZoneObservation(
            region_id=self._region.name,
            timestamp_s=now,
            state_version=self._actuator.state_version,
            rings=tuple(rings),
            interval_energy_kwh=sum(energy_kwh_by_ring.values()),
            interval_allocation_cost=sum(self._interval_alloc_by_ring.values()),
            interval_wait_req_s=self._interval_wait_req_s_total,
            interval_rejection_penalty_req_s=self._interval_rej_pen_total,
        )

    def observe(self, now: float) -> tuple[dict[str, dict[str, float | int]], np.ndarray]:
        """Costruisce l'osservazione strutturata e il vettore normalizzato allo stato corrente."""
        zone_obs = self.zone_observation(now)
        ring_obs = {ring.ring_id: ring.to_metrics_dict() for ring in zone_obs.rings}
        return ring_obs, self._encoder.peek(zone_obs)

    def close_interval(self, now: float, *, record_step: bool = True) -> IntervalSnapshot:
        """Chiude l'intervallo [interval_start_s, now], calcola K_t, D_t, reward e osservazione."""
        zone_obs = self.zone_observation(now)
        breakdown = self._reward_calc.compute(zone_obs)
        stacked_vec = self._encoder.push(zone_obs)
        ring_obs = {ring.ring_id: ring.to_metrics_dict() for ring in zone_obs.rings}

        energy_kwh = zone_obs.interval_energy_kwh
        allocation_cost = zone_obs.interval_allocation_cost
        wait_req_s = zone_obs.interval_wait_req_s
        rej_pen_s = zone_obs.interval_rejection_penalty_req_s

        # Aggiorna i totali cumulativi della replica.
        self.total_energy_kwh += energy_kwh
        self.total_allocation_cost += allocation_cost
        self.total_k_cost += breakdown.k_cost
        self.total_wait_req_s += wait_req_s
        self.total_rejection_penalty_req_s += rej_pen_s
        self.total_d_cost_req_s += breakdown.d_cost_req_s
        for ring in zone_obs.rings:
            self.total_energy_kwh_by_ring[ring.ring_id] += ring.interval_energy_kwh
            self.total_d_cost_by_ring[ring.ring_id] += ring.interval_delay_req_s
        if record_step:
            self.total_reward += breakdown.reward
            self.intervals_closed += 1

        snapshot = IntervalSnapshot(
            start_s=self._interval_start_s,
            end_s=now,
            energy_kwh=energy_kwh,
            allocation_cost=allocation_cost,
            k_cost=breakdown.k_cost,
            wait_req_s=wait_req_s,
            rejection_penalty_req_s=rej_pen_s,
            d_cost_req_s=breakdown.d_cost_req_s,
            reward=breakdown.reward,
            ring_observations=ring_obs,
            feature_vector=stacked_vec,
        )

        # Reset accumulatori di intervallo.
        self._interval_start_s = now
        for ring_id in self._actuator.ring_order:
            self._interval_energy_j_by_ring[ring_id] = 0.0
            self._interval_alloc_by_ring[ring_id] = 0.0
            self._interval_wait_req_s_by_ring[ring_id] = 0.0
            self._interval_rej_pen_by_ring[ring_id] = 0.0
            self._interval_arrivals_by_ring[ring_id] = 0
            self._interval_rejections_by_ring[ring_id] = 0
        self._interval_wait_req_s_total = 0.0
        self._interval_rej_pen_total = 0.0

        return snapshot

    def summary(self) -> dict:
        """Restituisce il riepilogo cumulativo delle metriche di costo e reward."""
        return {
            'intervals': self.intervals_closed,
            'total_energy_kwh': self.total_energy_kwh,
            'total_allocation_cost': self.total_allocation_cost,
            'total_k_cost': self.total_k_cost,
            'total_wait_req_s': self.total_wait_req_s,
            'total_rejection_penalty_req_s': self.total_rejection_penalty_req_s,
            'total_d_cost_req_s': self.total_d_cost_req_s,
            'total_reward': self.total_reward,
            'mean_reward': (self.total_reward / self.intervals_closed) if self.intervals_closed > 0 else None,
            'pending_requests': len(self._pending),
            'by_ring': {
                ring_id: {
                    'energy_kwh': self.total_energy_kwh_by_ring[ring_id],
                    'delay_req_s': self.total_d_cost_by_ring[ring_id],
                }
                for ring_id in self._actuator.ring_order
            },
        }
