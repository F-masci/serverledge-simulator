"""Adattatore dello scaler regionale per il simulatore (delega il core a ZoneController)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, Never, TYPE_CHECKING

from simulator.io.scaler import RingScalingConfig, ScalerConfig, ScalerMode
from simulator.model.events.scaler import ScalerStepEvent
from simulator.random.streams import RandomStreams, StreamKind
from simulator.scaler.contracts import ScalingCommand, ScalingPolicy, ScalingResult, ZoneObservation
from simulator.scaler.core import ZoneController

from .accounting import IntervalSnapshot, RegionAccounting
from .actuator import RegionActuator

if TYPE_CHECKING:
    from ray.rllib.core.rl_module.rl_module import RLModule
    from simulator.bindings import ActuatorBindings
    from simulator.core.simulation import SimulationContext
    from simulator.model.topology.region import Region
    from simulator.scaler.rllib import RLlibPPOPolicy


class Scaler:
    """
    Collega il `ZoneController` riusabile (core RL) al simulatore a eventi discreti.

    Tutta la logica di encoding, mascheramento, reward e policy vive in `ZoneController`
    e `ScalingPolicy`; questa classe gestisce solo il ciclo di eventi del simulatore.
    """

    mode: ClassVar[ScalerMode]

    def __init__(
        self,
        config: ScalerConfig,
        *,
        region: 'Region',
        policy: ScalingPolicy,
        ring_configs: Mapping[str, RingScalingConfig] | None = None,
        actuator_bindings: 'ActuatorBindings | None' = None,
        seed: int = 12345,
        stream_id: str | None = None,
    ) -> None:
        # Inizializza la configurazione, la regione e il generatore di numeri casuali.
        self._config = config
        self._region = region
        self._rng = RandomStreams(seed).stream(StreamKind.SCALING, stream_id or f'scaler/{region.name}')

        # Inizializza i componenti principali: actuator, accounting e controller.
        self._actuator = RegionActuator(region, ring_configs or {}, config.default_ring, rng=self._rng)
        if actuator_bindings is not None:
            actuator_bindings.bind(self._actuator)
        self._accounting = RegionAccounting(
            self._actuator,
            config.reward,
            interval_s=config.interval_s,
            history_length=config.history_length,
        )
        self._controller = ZoneController(
            region_id=region.name,
            ring_ids=self._actuator.ring_order,
            policy=policy,
            reward_config=config.reward,
            interval_s=config.interval_s,
            history_length=config.history_length,
            rng=self._rng,
        )

        # Inizializza lo stato interno dello scaler.
        self._attached = False
        self._finalized = False
        self._has_pending_interval = False

        # Inizializza il gestore di unsubscribe.
        self._unsubscribe: Callable[[], None] | None = None

        # Inizializza l'ultimo intervallo e l'ultimo risultato.
        self._last_interval: IntervalSnapshot | None = None
        self._last_result: ScalingResult | None = None

    @staticmethod
    def create(
        config: ScalerConfig,
        *,
        region: 'Region',
        ring_configs: Mapping[str, RingScalingConfig] | None = None,
        actuator_bindings: 'ActuatorBindings | None' = None,
        seed: int = 12345,
        stream_id: str | None = None,
        agent: 'RLlibPPOPolicy | RLModule | None' = None,
    ) -> 'Scaler | None':
        """Costruisce lo scaler corrispondente alla modalità configurata (None se disabilitato)."""

        # Se la modalità è NONE, non creare alcuno scaler
        if config.mode is ScalerMode.NONE:
            return None

        # Se la modalità è PPO, creare un'istanza di PPOScaler
        if config.mode is ScalerMode.PPO:
            return PPOScaler(
                config, region=region, ring_configs=ring_configs,
                actuator_bindings=actuator_bindings,
                seed=seed, stream_id=stream_id, agent=agent,
            )

        # Se la modalità è DDQN o RAINBOW, creare un'istanza di _UnavailableScaler
        raise NotImplementedError(f'Scaler {config.mode} non ancora disponibile nel simulatore')

    ### Components ###

    @property
    def actuator(self) -> RegionActuator:
        return self._actuator

    @property
    def accounting(self) -> RegionAccounting:
        return self._accounting

    @property
    def controller(self) -> ZoneController:
        """Espone il `ZoneController` riusabile (condiviso con serverledge-infrascaler)."""
        return self._controller

    ### Lifecycle ###

    def attach(self, ctx: 'SimulationContext', *, schedule_steps: bool = True) -> None:
        """Collega la contabilità all'observer della replica e avvia il ciclo periodico."""
        if self._attached:
            raise RuntimeError(f'Scaler della regione {self._region.name} già collegato')
        self._unsubscribe = ctx.observer.subscribe(
            self._accounting.on_notification, propagate_errors=True,
        )
        self._attached = True
        if schedule_steps:
            ctx.engine.schedule(ctx.now, ScalerStepEvent(ctx=ctx, region=self._region))

    def execute_command(
        self,
        command: ScalingCommand,
        ctx: 'SimulationContext',
        *,
        interval: IntervalSnapshot | None = None,
    ) -> ScalingResult:
        """Attua un comando sul simulatore e pubblica la notifica `ScalerDecision`."""
        self._accounting.advance_to(ctx.now)
        result = self._actuator.execute(command, ctx.now)
        self._last_result = result
        self._has_pending_interval = True

        ctx.observer.notify_scaler_decision(
            self._region,
            decision_id=command.decision_id,
            algorithm=self.mode.value,
            action=command.action.value,
            ring_id=command.ring_id,
            outcome=result.outcome.value,
            node_name=result.node_name,
            reward=interval.reward if interval is not None else None,
            k_cost=interval.k_cost if interval is not None else None,
            d_cost_req_s=interval.d_cost_req_s if interval is not None else None,
            energy_kwh=interval.energy_kwh if interval is not None else None,
        )
        return result

    def step(self, ctx: 'SimulationContext') -> ScalingResult:
        """Tick periodico: chiude l'intervallo, delega la decisione a ZoneController e attua."""
        now = ctx.now
        interval: IntervalSnapshot | None = None
        if self._has_pending_interval:
            zone_obs: ZoneObservation = self.accounting.zone_observation(now)
            interval = self.accounting.close_interval(now, record_step=True)
            self._last_interval = interval
            command, _ = self.controller.decide(zone_obs, record_history=True)
        else:
            zone_obs = self.accounting.zone_observation(now)
            command, _ = self.controller.decide(zone_obs, record_history=False)

        result = self.execute_command(command, ctx, interval=interval)
        ctx.engine.schedule_after(self._config.interval_s, ScalerStepEvent(ctx=ctx, region=self._region))
        return result

    def on_node_provisioned(self, ring_id: str, op_id: str, now: float) -> None:
        """Gestisce il completamento o fallimento del boot di un nodo."""
        self._accounting.advance_to(now)
        self._actuator.complete_provisioning(ring_id, op_id)

    def on_node_drained(self, ring_id: str, op_id: str, now: float) -> None:
        """Gestisce il rilascio definitivo di un nodo al termine del drain."""
        self._accounting.advance_to(now)
        node = self._actuator.complete_drain(ring_id, op_id)
        if node is not None:
            self._accounting.on_node_drain_completed(node.key, now)

    def finalize(self, now: float, *, truncated: bool = True) -> None:
        """Chiude l'eventuale intervallo residuo al termine della replica."""
        if self._finalized:
            return
        self._finalized = True

        if self._has_pending_interval and now > self._accounting._interval_start_s:
            self._last_interval = self._accounting.close_interval(now, record_step=True)
        else:
            self._accounting.advance_to(now)

        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    ### Metrics & Serialization ###

    def _algorithm_metrics(self) -> dict[str, Any]:
        return {}

    def metrics_snapshot(self, now: float | None = None) -> dict:
        """Restituisce lo snapshot completo delle metriche dello scaler regionale."""
        if now is not None and not self._finalized:
            self._accounting.advance_to(now)
        return {
            'mode': self.mode.value,
            'region': self._region.name,
            'interval_s': self._config.interval_s,
            'decisions': self._controller.decisions_count,
            'costs': self._accounting.summary(),
            'actuator': self._actuator.summary(),
            'algorithm': self._algorithm_metrics(),
        }

    def to_dict(self) -> dict:
        return self.metrics_snapshot()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __str__(self) -> str:
        return f'{type(self).__name__}(region={self._region.name}, mode={self.mode.value})'

    def __repr__(self) -> str:
        return f'<{type(self).__name__} region={self._region.name} mode={self.mode.value}>'


class PPOScaler(Scaler):
    """Scaler regionale basato su Ray RLlib PPO (PyTorch RLModule con action masking)."""

    mode = ScalerMode.PPO

    def __init__(
        self,
        config: ScalerConfig,
        *,
        region: 'Region',
        ring_configs: Mapping[str, RingScalingConfig] | None = None,
        actuator_bindings: 'ActuatorBindings | None' = None,
        seed: int = 12345,
        stream_id: str | None = None,
        agent: 'RLlibPPOPolicy | RLModule | None' = None,
    ) -> None:
        from simulator.scaler.contracts import FEATURES_PER_RING
        from simulator.scaler.rllib import RLlibPPOPolicy

        infra = region.infrastructure
        assert infra is not None
        num_rings = len(infra.rings)
        obs_dim = num_rings * FEATURES_PER_RING * config.history_length
        action_dim = 1 + 2 * num_rings

        if isinstance(agent, RLlibPPOPolicy):
            if agent.obs_dim != obs_dim or agent.action_dim != action_dim:
                raise ValueError(
                    f'Policy RLlib incompatibile: atteso ({obs_dim}, {action_dim}), '
                    f'ricevuto ({agent.obs_dim}, {agent.action_dim})'
                )
            policy = agent
        else:
            policy = RLlibPPOPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config.ppo,
                module=agent,
                seed=seed,
            )

        self._policy = policy
        super().__init__(
            config, region=region, policy=policy, ring_configs=ring_configs,
            actuator_bindings=actuator_bindings,
            seed=seed, stream_id=stream_id,
        )

    @property
    def policy(self) -> 'RLlibPPOPolicy':
        """Espone il wrapper di inferenza RLlib sottostante."""
        return self._policy

    def _algorithm_metrics(self) -> dict[str, Any]:
        return {
            'backend': 'ray.rllib',
            'deterministic': self._config.ppo.deterministic,
            'checkpoint_path': self._config.ppo.checkpoint_path,
        }


class _UnavailableScaler(Scaler):
    """Segnaposto per algoritmi RL previsti nella tesi ma non ancora implementati."""

    def __init__(self) -> None:
        self._raise_unavailable()

    def _raise_unavailable(self) -> Never:
        raise NotImplementedError(f'Scaler {self.mode} non ancora disponibile nel simulatore')


class DDQNScaler(_UnavailableScaler):
    mode = ScalerMode.DDQN


class RainbowScaler(_UnavailableScaler):
    mode = ScalerMode.RAINBOW
