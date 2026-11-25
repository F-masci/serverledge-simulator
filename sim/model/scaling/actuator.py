"""Attuatore di scaling per regione e per ring (ruolo Infra Controller della rev-1)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import Generator

from sim.model.events.scaler import NodeDrainedEvent, NodeProvisionedEvent
from sim.model.node import Node
from sim.model.offloading import NodeScheduling
from sim.model.ring import ArchitectureRing
from sim.random.streams import RandomStreams, StreamKind
from sim.io.scaler import RingScalingConfig
from sim.scaler.contracts import (
    CommandOutcome, ScalingActionKind, ScalingCommand, ScalingResult,
)
from sim.scaler.core import ActionCodec

if TYPE_CHECKING:
    from sim.core.simulation import SimulationContext
    from sim.model.region import Region


@dataclass
class ProvisioningOp:
    """Operazione di provisioning/avvio di un nodo in corso su un ring."""

    op_id: str
    ring_id: str
    node_name: str
    started_at_s: float
    ready_at_s: float
    power_w: float


@dataclass
class DrainingOp:
    """Operazione di drain/spegnimento di un nodo rimosso dal routing di un ring."""

    op_id: str
    ring_id: str
    node: Node
    started_at_s: float
    drain_completed_at_s: float | None = None
    shutdown_scheduled: bool = False


class RingActuator:
    """Gestisce quota, provisioning, selezione deterministica e drain di un singolo ring."""

    def __init__(
        self,
        ring: ArchitectureRing,
        config: RingScalingConfig,
    ) -> None:
        self._ring = ring
        initial_nodes = len(ring)
        resolved_max = max(initial_nodes, 1) if config.max_nodes is None else config.max_nodes
        if config.min_nodes > resolved_max:
            raise ValueError(
                f'Ring {ring.id}: min_nodes ({config.min_nodes}) > max_nodes ({resolved_max})'
            )
        if initial_nodes > resolved_max:
            raise ValueError(
                f'Ring {ring.id}: nodi iniziali ({initial_nodes}) > max_nodes ({resolved_max})'
            )

        self._config = config
        self._min_nodes = config.min_nodes
        self._max_nodes = resolved_max
        self._node_seq = count(1)
        self._op_seq = count(1)

        # Operazioni asincrone in corso che riservano quota e consumano energia.
        self._provisioning: dict[str, ProvisioningOp] = {}
        self._draining: dict[str, DrainingOp] = {}
        self._draining_by_node: dict[str, str] = {}
        self._last_transition_s: float = -float('inf')

        # Contatori cumulativi delle transizioni sul ring.
        self.scale_out_requested = 0
        self.scale_out_completed = 0
        self.scale_out_failed = 0
        self.scale_in_requested = 0
        self.scale_in_completed = 0

    ### Properties ###

    @property
    def ring(self) -> ArchitectureRing:
        return self._ring

    @property
    def ring_id(self) -> str:
        return self._ring.id

    @property
    def config(self) -> RingScalingConfig:
        return self._config

    @property
    def min_nodes(self) -> int:
        return self._min_nodes

    @property
    def max_nodes(self) -> int:
        return self._max_nodes

    @property
    def ready_count(self) -> int:
        """Numero di nodi pronti e attivi nel routing del ring."""
        return len(self._ring)

    @property
    def provisioning_count(self) -> int:
        """Numero di nodi attualmente in acquisizione/boot."""
        return len(self._provisioning)

    @property
    def draining_count(self) -> int:
        """Numero di nodi attualmente in drain."""
        return len(self._draining)

    @property
    def allocated_count(self) -> int:
        """Nodi che occupano quota: pronti + in acquisizione/boot + in drain."""
        return self.ready_count + self.provisioning_count + self.draining_count

    @property
    def remaining_quota(self) -> int:
        """Quota residua acquisibile sul ring."""
        return max(0, self._max_nodes - self.allocated_count)

    @property
    def provisioning_ops(self) -> tuple[ProvisioningOp, ...]:
        return tuple(self._provisioning.values())

    @property
    def draining_ops(self) -> tuple[DrainingOp, ...]:
        return tuple(self._draining.values())

    def estimated_remaining_boot_s(self, now: float) -> float:
        """Tempo residuo massimo stimato tra le operazioni di boot in corso (0 se assenti)."""
        if not self._provisioning:
            return 0.0
        return max(max(0.0, op.ready_at_s - now) for op in self._provisioning.values())

    ### Feasibility & Selection ###

    def _cooldown_ready(self, now: float) -> bool:
        if self._config.cooldown_s <= 0.0:
            return True
        return (now - self._last_transition_s) >= self._config.cooldown_s

    def can_scale_out(self, now: float) -> bool:
        """Verifica se è possibile avviare un nuovo nodo entro la quota e il cooldown."""
        return self.remaining_quota > 0 and self._cooldown_ready(now)

    def can_scale_in(self, now: float) -> bool:
        """Verifica se esiste almeno un nodo eleggibile al consolidamento sopra min_nodes."""
        return self.ready_count > self._min_nodes and self._cooldown_ready(now)

    def select_scale_in_candidate(self) -> Node | None:
        """Selezione deterministica del nodo da consolidare: minimo carico corrente, tie-break per nome."""
        nodes = tuple(self._ring)
        if not nodes:
            return None
        return min(
            nodes,
            key=lambda n: (
                n.active_requests_count + n.queued_requests_count,
                n.resources.used_cpus,
                n.resources.busy_pool_used_mem,
                n.resources.warm_pool_used_mem,
                n.key,
            ),
        )

    ### Operations ###

    def start_scale_out(self, now: float, *, existing_names: set[str]) -> ProvisioningOp | None:
        """Prenota quota e crea un'operazione di provisioning asincrona."""
        if not self.can_scale_out(now):
            return None

        # Genera un nome univoco nella regione che non collida con nodi esistenti o in transizione.
        while True:
            node_name = f'{self.ring_id}-auto-{next(self._node_seq)}'
            if node_name not in existing_names:
                break

        op_id = f'prov-{self.ring_id}-{next(self._op_seq)}'
        op = ProvisioningOp(
            op_id=op_id,
            ring_id=self.ring_id,
            node_name=node_name,
            started_at_s=now,
            ready_at_s=now + self._config.boot_time_s,
            power_w=self._config.effective_boot_power_w,
        )
        self._provisioning[op_id] = op
        self._last_transition_s = now
        self.scale_out_requested += 1
        return op

    def pop_provisioning(self, op_id: str) -> ProvisioningOp | None:
        """Estrae l'operazione di provisioning completata o fallita."""
        return self._provisioning.pop(op_id, None)

    def start_scale_in(self, now: float) -> DrainingOp | None:
        """Seleziona il nodo candidato, lo rimuove dal routing e avvia il drain."""
        if not self.can_scale_in(now):
            return None
        candidate = self.select_scale_in_candidate()
        if candidate is None:
            return None

        self._ring.infrastructure.remove_node(self.ring_id, candidate.key)
        op_id = f'drain-{self.ring_id}-{next(self._op_seq)}'
        op = DrainingOp(
            op_id=op_id,
            ring_id=self.ring_id,
            node=candidate,
            started_at_s=now,
        )
        self._draining[op_id] = op
        self._draining_by_node[candidate.key] = op_id
        self._last_transition_s = now
        self.scale_in_requested += 1
        return op

    def get_draining_for_node(self, node_key: str) -> DrainingOp | None:
        """Restituisce l'operazione di drain attiva per il nodo, se presente."""
        op_id = self._draining_by_node.get(node_key)
        return None if op_id is None else self._draining.get(op_id)

    def pop_draining(self, op_id: str) -> DrainingOp | None:
        """Rimuove e restituisce un'operazione di drain conclusa."""
        op = self._draining.pop(op_id, None)
        if op is not None:
            self._draining_by_node.pop(op.node.key, None)
        return op


class RegionActuator:
    """Coordina gli attuatori di tutti i ring di una regione e lo spazio delle azioni."""

    def __init__(
        self,
        region: 'Region',
        ring_configs: Mapping[str, RingScalingConfig],
        default_config: RingScalingConfig,
        *,
        rng: Generator,
    ) -> None:
        infrastructure = region.infrastructure
        if infrastructure is None:
            raise ValueError(f'La regione {region.name} non dispone di un\'infrastruttura simulata')

        self._region = region
        self._rng = rng
        self._state_version = 0
        self._ring_order: tuple[str, ...] = tuple(infrastructure.rings.keys())
        self._rings: dict[str, RingActuator] = {
            ring_id: RingActuator(
                ring,
                ring_configs.get(ring_id, default_config),
            )
            for ring_id, ring in infrastructure.rings.items()
        }

        # Codec condiviso per spazio azioni (1 + 2P) e maschere.
        self._codec = ActionCodec(self._ring_order)
        self._action_space = tuple(
            self._codec.decode(i) for i in range(self._codec.action_dim)
        )

        # Contatori regionali dei comandi.
        self.hold_actions = 0
        self.rejected_commands = 0

    ### Properties ###

    @property
    def region(self) -> 'Region':
        return self._region

    @property
    def ring_order(self) -> tuple[str, ...]:
        return self._ring_order

    @property
    def rings(self) -> Mapping[str, RingActuator]:
        return self._rings

    @property
    def action_dim(self) -> int:
        return len(self._action_space)

    @property
    def state_version(self) -> int:
        return self._state_version

    def decode_action(self, action_index: int) -> tuple[ScalingActionKind, str | None]:
        """Decodifica un indice discreto in (ScalingActionKind, ring_id | None)."""
        return self._action_space[action_index]

    def action_mask(self, now: float) -> np.ndarray:
        """Restituisce la maschera booleana delle azioni ammissibili al tempo now."""
        mask = np.zeros(len(self._action_space), dtype=bool)
        mask[0] = True  # Mantieni resta sempre disponibile.
        for idx, (kind, ring_id) in enumerate(self._action_space[1:], start=1):
            assert ring_id is not None
            ring_act = self._rings[ring_id]
            if kind is ScalingActionKind.SCALE_OUT:
                mask[idx] = ring_act.can_scale_out(now)
            elif kind is ScalingActionKind.SCALE_IN:
                mask[idx] = ring_act.can_scale_in(now)
        return mask

    def _all_known_node_names(self) -> set[str]:
        """Raccoglie tutti i nomi di nodi attivi, in provisioning o in drain nella regione."""
        infra = self._region.infrastructure
        assert infra is not None
        names = {node.key for ring in infra.rings.values() for node in ring}
        for ring_act in self._rings.values():
            names.update(op.node_name for op in ring_act.provisioning_ops)
            names.update(op.node.key for op in ring_act.draining_ops)
        return names

    ### Command Execution ###

    def execute(self, command: ScalingCommand, ctx: 'SimulationContext') -> ScalingResult:
        """Verifica l'eseguibilità allo stato corrente e attua il comando senza sostituzioni silenziose."""
        now = ctx.now
        if command.action is ScalingActionKind.HOLD:
            self.hold_actions += 1
            return ScalingResult(command=command, outcome=CommandOutcome.COMPLETED, completed_at_s=now)

        if command.ring_id is None or command.ring_id not in self._rings:
            self.rejected_commands += 1
            return ScalingResult(
                command=command,
                outcome=CommandOutcome.REJECTED,
                reason='unknown_ring',
                completed_at_s=now,
            )

        ring_act = self._rings[command.ring_id]
        if command.action is ScalingActionKind.SCALE_OUT:
            op = ring_act.start_scale_out(now, existing_names=self._all_known_node_names())
            if op is None:
                self.rejected_commands += 1
                return ScalingResult(
                    command=command,
                    outcome=CommandOutcome.REJECTED,
                    reason='quota_or_cooldown',
                    completed_at_s=now,
                )
            self._state_version += 1
            event = NodeProvisionedEvent(
                ctx=ctx,
                region=self._region,
                ring_id=ring_act.ring_id,
                op_id=op.op_id,
            )
            if ring_act.config.boot_time_s == 0.0:
                event.execute()
                return ScalingResult(
                    command=command,
                    outcome=CommandOutcome.COMPLETED,
                    node_name=op.node_name,
                    completed_at_s=now,
                )
            ctx.engine.schedule_after(ring_act.config.boot_time_s, event)
            return ScalingResult(
                command=command,
                outcome=CommandOutcome.IN_PROGRESS,
                node_name=op.node_name,
            )

        if command.action is ScalingActionKind.SCALE_IN:
            op = ring_act.start_scale_in(now)
            if op is None:
                self.rejected_commands += 1
                return ScalingResult(
                    command=command,
                    outcome=CommandOutcome.REJECTED,
                    reason='no_eligible_node_or_cooldown',
                    completed_at_s=now,
                )
            self._state_version += 1
            self._maybe_schedule_drain_completion(ring_act, op, ctx)
            return ScalingResult(
                command=command,
                outcome=(
                    CommandOutcome.COMPLETED
                    if op.op_id not in ring_act._draining
                    else CommandOutcome.IN_PROGRESS
                ),
                node_name=op.node.key,
                completed_at_s=now if op.op_id not in ring_act._draining else None,
            )

        self.rejected_commands += 1
        return ScalingResult(
            command=command,
            outcome=CommandOutcome.REJECTED,
            reason='unsupported_action',
            completed_at_s=now,
        )

    def complete_provisioning(self, ring_id: str, op_id: str, ctx: 'SimulationContext') -> Node | None:
        """Completa (o fa fallire) un'operazione di provisioning al termine di boot_time_s."""
        ring_act = self._rings.get(ring_id)
        if ring_act is None:
            return None
        op = ring_act.pop_provisioning(op_id)
        if op is None:
            return None

        # Verifica eventuale fallimento del provisioning del provider.
        fail_prob = ring_act.config.provision_failure_prob
        if fail_prob > 0.0 and float(self._rng.random()) < fail_prob:
            ring_act.scale_out_failed += 1
            self._state_version += 1
            return None

        # Crea e registra il nodo nel ring con gli stessi default dello scenario.
        scenario = ctx.scenario
        infra = self._region.infrastructure
        assert infra is not None
        streams = RandomStreams(scenario.seed)
        eviction_rng = streams.stream(StreamKind.EVICTION, f'{self._region.name}/{ring_id}/{op.node_name}')
        node = infra.create_node(
            ring_id,
            op.node_name,
            idle_timeout=scenario.containers.idle_timeout,
            queue_capacity=scenario.scheduler.queue_capacity,
            eviction_rng=eviction_rng,
        )
        infra.configure_scheduler(
            node,
            NodeScheduling(
                policy=scenario.scheduler.policy,
                fallback_local=scenario.scheduler.fallback_local,
                offloading_cache_validity_s=scenario.scheduler.offloading_cache_validity_s,
            ),
        )
        ring_act.scale_out_completed += 1
        self._state_version += 1
        return node

    def on_node_activity_changed(self, node: Node, ctx: 'SimulationContext') -> None:
        """Controlla se un nodo in drain ha esaurito tutte le richieste in volo e in coda."""
        ring = node._ring
        if ring is None:
            return
        ring_act = self._rings.get(ring.id)
        if ring_act is None:
            return
        op = ring_act.get_draining_for_node(node.key)
        if op is None:
            return
        self._maybe_schedule_drain_completion(ring_act, op, ctx)

    def _maybe_schedule_drain_completion(
        self, ring_act: RingActuator, op: DrainingOp, ctx: 'SimulationContext',
    ) -> None:
        """Pianifica lo spegnimento del nodo non appena il drain è completo."""
        if op.shutdown_scheduled or not op.node.is_idle:
            return
        op.drain_completed_at_s = ctx.now
        op.shutdown_scheduled = True
        event = NodeDrainedEvent(
            ctx=ctx,
            region=self._region,
            ring_id=ring_act.ring_id,
            op_id=op.op_id,
        )
        if ring_act.config.shutdown_time_s == 0.0:
            event.execute()
        else:
            ctx.engine.schedule_after(ring_act.config.shutdown_time_s, event)

    def complete_drain(self, ring_id: str, op_id: str, ctx: 'SimulationContext') -> Node | None:
        """Spegne definitivamente un nodo drenato e rilascia la sua quota."""
        ring_act = self._rings.get(ring_id)
        if ring_act is None:
            return None
        op = ring_act.pop_draining(op_id)
        if op is None:
            return None

        if op.node.is_idle:
            op.node.shutdown()
        ring_act.scale_in_completed += 1
        self._state_version += 1
        return op.node

    ### Metrics Summary ###

    def summary(self) -> dict:
        """Restituisce il riepilogo delle operazioni di scaling per regione e per ring."""
        return {
            'state_version': self._state_version,
            'hold_actions': self.hold_actions,
            'rejected_commands': self.rejected_commands,
            'scale_out_requested': sum(r.scale_out_requested for r in self._rings.values()),
            'scale_out_completed': sum(r.scale_out_completed for r in self._rings.values()),
            'scale_out_failed': sum(r.scale_out_failed for r in self._rings.values()),
            'scale_in_requested': sum(r.scale_in_requested for r in self._rings.values()),
            'scale_in_completed': sum(r.scale_in_completed for r in self._rings.values()),
            'by_ring': {
                ring_id: {
                    'min_nodes': r.min_nodes,
                    'max_nodes': r.max_nodes,
                    'ready_nodes': r.ready_count,
                    'provisioning_nodes': r.provisioning_count,
                    'draining_nodes': r.draining_count,
                    'remaining_quota': r.remaining_quota,
                    'scale_out_requested': r.scale_out_requested,
                    'scale_out_completed': r.scale_out_completed,
                    'scale_out_failed': r.scale_out_failed,
                    'scale_in_requested': r.scale_in_requested,
                    'scale_in_completed': r.scale_in_completed,
                }
                for ring_id, r in self._rings.items()
            },
        }
