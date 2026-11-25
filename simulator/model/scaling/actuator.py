"""Attuatore di scaling per regione e per ring (ruolo Infra Controller della rev-1)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING

from numpy.random import Generator
from simulator.io.scaler import RingScalingConfig
from simulator.model.topology.node import Node
from simulator.model.topology.ring import ArchitectureRing
from simulator.scaler.contracts import (
    CommandOutcome, ScalingActionKind, ScalingCommand, ScalingResult,
)

if TYPE_CHECKING:
    from simulator.model.topology.infrastructure import Infrastructure
    from simulator.model.topology.region import Region


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
        # Riferimento al ring.
        self._ring = ring

        # Valida la configurazione di scaling rispetto al numero di nodi iniziali.
        #
        # La configurazione può avere max_nodes=None, che significa "nessun limite superiore"
        # e viene risolto al numero di nodi iniziali.
        initial_nodes = len(ring)
        resolved_max = max(initial_nodes, 1) if config.max_nodes is None else config.max_nodes
        if config.min_nodes > resolved_max:
            raise ValueError(f'Ring {ring.id}: min_nodes ({config.min_nodes}) > max_nodes ({resolved_max})')
        if initial_nodes > resolved_max:
            raise ValueError(f'Ring {ring.id}: nodi iniziali ({initial_nodes}) > max_nodes ({resolved_max})')

        # Inizializza lo stato dell'attuatore.
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

    ### Infrastructure ###

    @property
    def ring(self) -> ArchitectureRing:
        return self._ring

    @property
    def ring_id(self) -> str:
        return self.ring.id

    @property
    def infrastructure(self) -> Infrastructure:
        return self.ring.infrastructure

    ### Configuration ###

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

        # Calcola il tempo residuo massimo tra le operazioni di boot in corso.
        boot_remaining_s = [max(0.0, op.ready_at_s - now) for op in self._provisioning.values()]
        return max(boot_remaining_s) if boot_remaining_s else 0.0

    ###############################
    ### Feasibility & Selection ###
    ###############################

    def _cooldown_ready(self, now: float) -> bool:
        """Verifica se il cooldown è scaduto dall'ultima transizione di scaling."""
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

    ##################
    ### Operations ###
    ##################

    ## Scale-Out ##

    def start_scale_out(self, now: float, *, existing_names: set[str]) -> ProvisioningOp | None:
        """Prenota quota e crea un'operazione di provisioning asincrona."""
        if not self.can_scale_out(now):
            return None

        # Genera un nome univoco nella regione che non collida con nodi esistenti o in transizione.
        #
        # I nomi da considerare sono quelli dei nodi già pronti, in provisioning o in drain; i nodi
        # devono essere univoci a livello di regione, non solo di ring, per evitare collisioni con eventuali
        # nodi in altri ring della stessa regione.
        while True:
            node_name = f'{self.ring_id}-auto-{next(self._node_seq)}'
            if node_name not in existing_names:
                break

        # Crea l'operazione di provisioning e la registra nello stato dell'attuatore.
        op_id = f'prov-{self.ring_id}-{next(self._op_seq)}'
        op = ProvisioningOp(
            op_id=op_id,
            ring_id=self.ring_id,
            node_name=node_name,
            started_at_s=now,
            ready_at_s=now + self.config.boot_time_s,
            power_w=self.config.effective_boot_power_w,
        )
        self._provisioning[op_id] = op

        # Aggiorna il timestamp dell'ultima transizione
        # e incrementa il contatore delle richieste di scale-out.
        self._last_transition_s = now
        self.scale_out_requested += 1

        return op

    def is_provisioning(self, op_id: str) -> bool:
        """Indica se l'operazione di provisioning è ancora in corso."""
        return op_id in self._provisioning

    def pop_provisioning(self, op_id: str) -> ProvisioningOp | None:
        """Estrae l'operazione di provisioning completata o fallita."""
        return self._provisioning.pop(op_id, None)

    ## Scale-In ##

    def start_scale_in(self, now: float) -> DrainingOp | None:
        """Seleziona il nodo candidato, lo rimuove dal routing e avvia il drain."""
        if not self.can_scale_in(now):
            return None

        # Seleziona il nodo candidato per lo scale-in.
        candidate = self.select_scale_in_candidate()
        if candidate is None:
            return None

        # Rimuove il nodo dal routing del ring e registra l'operazione di drain.
        self.infrastructure.remove_node(self.ring_id, candidate.key)
        op_id = f'drain-{self.ring_id}-{next(self._op_seq)}'
        op = DrainingOp(
            op_id=op_id,
            ring_id=self.ring_id,
            node=candidate,
            started_at_s=now,
        )
        self._draining[op_id] = op
        self._draining_by_node[candidate.key] = op_id

        # Aggiorna il timestamp dell'ultima transizione e incrementa il contatore delle richieste di scale-in.
        self._last_transition_s = now
        self.scale_in_requested += 1

        return op

    def is_draining(self, op_id: str) -> bool:
        """Indica se l'operazione di drain è ancora in corso."""
        return op_id in self._draining

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

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile dello stato di scaling del ring."""
        return (
            f'RingActuator(ring={self.ring_id}, ready={self.ready_count}, '
            f'provisioning={self.provisioning_count}, draining={self.draining_count}, '
            f'bounds=[{self._min_nodes}, {self._max_nodes}])'
        )

    def to_dict(self) -> dict:
        """Restituisce lo stato e i contatori di scaling del ring in formato serializzabile."""
        return {
            'min_nodes': self._min_nodes,
            'max_nodes': self._max_nodes,
            'ready_nodes': self.ready_count,
            'provisioning_nodes': self.provisioning_count,
            'draining_nodes': self.draining_count,
            'remaining_quota': self.remaining_quota,
            'scale_out_requested': self.scale_out_requested,
            'scale_out_completed': self.scale_out_completed,
            'scale_out_failed': self.scale_out_failed,
            'scale_in_requested': self.scale_in_requested,
            'scale_in_completed': self.scale_in_completed,
        }

    def to_json(self) -> str:
        """Serializza lo stato dell'attuatore del ring in formato JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return (
            f'<RingActuator ring={self.ring_id} ready={self.ready_count} '
            f'provisioning={self.provisioning_count} draining={self.draining_count}>'
        )


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
        # Verifica che la regione abbia un'infrastruttura simulata.
        infrastructure = region.infrastructure
        if infrastructure is None:
            raise ValueError(f'La regione {region.name} non dispone di un\'infrastruttura simulata')

        # Inizializza lo stato dell'attuatore regionale.
        #
        # L'ordine dei ring è stabile e deterministico,
        # basato sull'ordine di inserimento dei ring nell'infrastruttura.
        #
        # Ogni ring ha un proprio attuatore che gestisce quota, provisioning e drain.
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

        # Contatori regionali dei comandi.
        self.hold_actions = 0
        self.rejected_commands = 0

        ### Callback (bindate da Simulation tramite ActuatorBindings) ###
        self.on_provision_started: Callable[[Region, str, ProvisioningOp, float], None] | None = None
        self.on_provision_completed: Callable[[Region, str, str], Node] | None = None
        self.on_drain_ready: Callable[[Region, str, DrainingOp, float], None] | None = None

    ### Infrastructure ###

    @property
    def region(self) -> Region:
        return self._region

    @property
    def ring_order(self) -> tuple[str, ...]:
        return self._ring_order

    @property
    def rings(self) -> Mapping[str, RingActuator]:
        return self._rings

    def _all_known_node_names(self) -> set[str]:
        """Raccoglie tutti i nomi di nodi attivi, in provisioning o in drain nella regione."""
        infra = self._region.infrastructure
        assert infra is not None
        names = {node.key for ring in infra.rings.values() for node in ring}
        for ring_act in self._rings.values():
            names.update(op.node_name for op in ring_act.provisioning_ops)
            names.update(op.node.key for op in ring_act.draining_ops)
        return names

    @property
    def state_version(self) -> int:
        """Versione incrementale dello stato dell'attuatore regionale, aggiornata ad ogni transizione di scaling."""
        return self._state_version

    ##################
    ### Operations ###
    ##################

    def execute(self, command: ScalingCommand, now: float) -> ScalingResult:
        """Verifica l'eseguibilità allo stato corrente e attua il comando senza sostituzioni silenziose."""

        # --- Percorso 1: HOLD (nessuna modifica all'infrastruttura) ---
        if command.action is ScalingActionKind.HOLD:
            self.hold_actions += 1
            return ScalingResult(
                command=command,
                outcome=CommandOutcome.COMPLETED,
                completed_at_s=now,
            )

        # --- Validazione del ring destinatario ---
        if command.ring_id is None or command.ring_id not in self._rings:
            self.rejected_commands += 1
            return ScalingResult(
                command=command,
                outcome=CommandOutcome.REJECTED,
                reason='unknown_ring',
                completed_at_s=now,
            )

        ring_act = self._rings[command.ring_id]

        # --- Percorso 2: SCALE_OUT (avvio provisioning di un nuovo nodo) ---
        if command.action is ScalingActionKind.SCALE_OUT:
            return self._execute_scale_out(command, ring_act, now)

        # --- Percorso 3: SCALE_IN (rimozione dal routing e avvio drain) ---
        if command.action is ScalingActionKind.SCALE_IN:
            return self._execute_scale_in(command, ring_act, now)

    ## Scale-Out ##

    def _execute_scale_out(
        self, command: ScalingCommand, ring_act: RingActuator, now: float,
    ) -> ScalingResult:
        """Avvia il provisioning di un nuovo nodo sul ring e notifica la simulazione."""
        op = ring_act.start_scale_out(now, existing_names=self._all_known_node_names())
        if op is None:
            self.rejected_commands += 1
            return ScalingResult(
                command=command,
                outcome=CommandOutcome.REJECTED,
                reason='quota_or_cooldown',
                completed_at_s=now,
            )

        # La prenotazione della quota modifica lo stato infrastrutturale.
        self._state_version += 1

        # Delega alla simulazione lo scheduling (o il completamento immediato) del boot.
        if self.on_provision_started is not None:
            self.on_provision_started(self._region, ring_act.ring_id, op, ring_act.config.boot_time_s)

        # Se boot_time_s == 0 e la callback ha completato il boot in modo sincrono,
        # l'operazione non è più tra quelle in corso.
        completed_sync = not ring_act.is_provisioning(op.op_id)
        return ScalingResult(
            command=command,
            outcome=CommandOutcome.COMPLETED if completed_sync else CommandOutcome.IN_PROGRESS,
            node_name=op.node_name,
            completed_at_s=now if completed_sync else None,
        )

    def complete_provisioning(self, ring_id: str, op_id: str) -> Node | None:
        """Completa (o fa fallire) un'operazione di provisioning al termine di boot_time_s."""

        # Recupera il ring e l'operazione pendente; ignora notifiche duplicate o per ring sconosciuti.
        ring_act = self._rings.get(ring_id)
        if ring_act is None:
            return None

        op = ring_act.pop_provisioning(op_id)
        if op is None:
            return None

        # --- Percorso 1: fallimento stocastico del provisioning del provider ---
        fail_prob = ring_act.config.provision_failure_prob
        if fail_prob > 0.0 and float(self._rng.random()) < fail_prob:
            ring_act.scale_out_failed += 1
            self._state_version += 1
            return None

        # --- Percorso 2: boot completato, delega la creazione del nodo alla simulazione ---
        if self.on_provision_completed is None:
            raise RuntimeError(f'Callback on_provision_completed non configurata sull\'attuatore della regione {self._region.name}')
        node = self.on_provision_completed(self._region, ring_id, op.node_name)

        # Aggiorna contatori e versione dello stato infrastrutturale.
        ring_act.scale_out_completed += 1
        self._state_version += 1

        return node

    ## Scale-In ##

    def _execute_scale_in(
        self, command: ScalingCommand, ring_act: RingActuator, now: float,
    ) -> ScalingResult:
        """Rimuove dal routing il nodo candidato del ring e avvia il drain fino allo spegnimento."""
        op = ring_act.start_scale_in(now)
        if op is None:
            self.rejected_commands += 1
            return ScalingResult(
                command=command,
                outcome=CommandOutcome.REJECTED,
                reason='no_eligible_node_or_cooldown',
                completed_at_s=now,
            )

        # La rimozione dal routing modifica lo stato infrastrutturale.
        self._state_version += 1

        # Se il nodo è già idle, notifica subito la simulazione per lo spegnimento;
        # altrimenti attende che si svuotino richieste in volo e in coda.
        self._notify_if_drain_ready(ring_act, op, now)

        # Se il nodo era già idle e shutdown_time_s == 0, il drain si è già concluso.
        completed_sync = not ring_act.is_draining(op.op_id)
        return ScalingResult(
            command=command,
            outcome=CommandOutcome.COMPLETED if completed_sync else CommandOutcome.IN_PROGRESS,
            node_name=op.node.key,
            completed_at_s=now if completed_sync else None,
        )

    def _notify_if_drain_ready(
        self, ring_act: RingActuator, op: DrainingOp, now: float,
    ) -> None:
        """Verifica se il nodo in drain è idle e notifica la simulazione per lo spegnimento."""

        # Se lo spegnimento è già stato avviato o il nodo ha ancora richieste attive/in coda, non fa nulla.
        if op.shutdown_scheduled or not op.node.is_idle:
            return

        # Registra l'istante di fine drain e marca lo spegnimento come già notificato (idempotenza).
        op.drain_completed_at_s = now
        op.shutdown_scheduled = True

        # Delega alla simulazione lo scheduling (o l'esecuzione immediata) dello spegnimento definitivo.
        if self.on_drain_ready is not None:
            self.on_drain_ready(self._region, ring_act.ring_id, op, ring_act.config.shutdown_time_s)

    def check_node_drain(self, node: Node, now: float) -> None:
        """Verifica se un nodo in drain ha esaurito tutte le richieste in volo e in coda."""

        # Ignora nodi non associati a un ring gestito da questo attuatore.
        ring_act = self._rings.get(node.ring.id)
        if ring_act is None:
            return

        # Se il nodo ha un drain attivo ed è diventato idle, notifica la simulazione.
        op = ring_act.get_draining_for_node(node.key)
        if op is None:
            return

        self._notify_if_drain_ready(ring_act, op, now)

    def complete_drain(self, ring_id: str, op_id: str) -> Node | None:
        """Spegne definitivamente un nodo drenato e rilascia la sua quota."""

        # Recupera il ring e l'operazione di drain pendente; ignora notifiche duplicate o sconosciute.
        ring_act = self._rings.get(ring_id)
        if ring_act is None:
            return None

        op = ring_act.pop_draining(op_id)
        if op is None:
            return None

        # Spegne definitivamente il nodo (ormai idle) rilasciandone risorse e container residui.
        if op.node.is_idle:
            op.node.shutdown()

        # Aggiorna contatori e versione dello stato infrastrutturale (quota rilasciata).
        ring_act.scale_in_completed += 1
        self._state_version += 1

        return op.node

    #####################
    ### Serialization ###
    #####################

    ## Metrics Summary ##

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
                ring_id: r.to_dict()
                for ring_id, r in self._rings.items()
            },
        }

    ## Presentation ##

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile dell'attuatore regionale."""
        return (
            f'RegionActuator(region={self._region.name}, rings={len(self._rings)}, '
            f'state_version={self._state_version})'
        )

    def to_dict(self) -> dict:
        """Restituisce lo stato e le metriche dell'attuatore regionale in formato serializzabile."""
        return self.summary()

    def to_json(self) -> str:
        """Serializza lo stato dell'attuatore regionale in formato JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return (
            f'<RegionActuator region={self._region.name} rings={len(self._rings)} '
            f'state_version={self._state_version}>'
        )
