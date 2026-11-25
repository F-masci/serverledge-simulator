"""Ambiente Gymnasium per l'addestramento RL (Ray RLlib) dello scaler regionale."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from sim.io.scenario import Scenario
from sim.model.events import Event
from sim.io.scaler import ScalerConfig, ScalerMode
from sim.scaler.contracts import FEATURES_PER_RING, ScalingCommand

if TYPE_CHECKING:
    from sim.core.simulation import Simulation


@dataclass(frozen=True)
class _StepBarrierEvent(Event):
    """Evento barriera che arresta il motore esattamente al termine dell'intervallo Delta."""

    reached: list[bool]

    def execute(self) -> None:
        self.reached[0] = True


class ServerledgeScalingEnv(gym.Env):
    """
    Ambiente Gymnasium a eventi discreti per l'autoscaling di una singola regione.

    Espone:
      - observation_space: Dict con 'observations' (Box float32) e 'action_mask' (Box 0/1 float32)
      - action_space: Discrete(1 + 2 * num_rings) -> hold + (scale_out, scale_in) per ring
      - reward: r_t = -[w_K * K_t / K_ref + w_T * D_t / D_ref] calcolata su [t, t + Delta]

    Compatibile direttamente con Ray RLlib tramite `env_config`.
    """

    metadata = {'render_modes': []}

    def __init__(self, env_config: Mapping[str, Any] | None = None) -> None:
        super().__init__()
        cfg = dict(env_config or {})
        raw_scenario = cfg.get('scenario')
        if raw_scenario is None:
            raise ValueError("env_config richiede 'scenario' (Scenario o percorso file)")

        if isinstance(raw_scenario, Scenario):
            self._base_scenario = raw_scenario
        else:
            self._base_scenario = Scenario.load(Path(raw_scenario))

        self._region_name: str = str(cfg.get('region') or self._base_scenario.primary_region)
        if self._region_name not in self._base_scenario.regions:
            raise ValueError(f'Regione sconosciuta nello scenario: {self._region_name}')

        region_cfg = self._base_scenario.regions[self._region_name]
        if region_cfg.infrastructure is None or not region_cfg.infrastructure.rings:
            raise ValueError(f'La regione {self._region_name} deve avere almeno un ring simulato')

        # Garantisce che lo scaler della regione sia configurato (se disabilitato, usa PPO di default).
        scaler_cfg = region_cfg.scaler if region_cfg.scaler is not None else self._base_scenario.scaler
        if not scaler_cfg.is_enabled:
            scaler_cfg = ScalerConfig(mode=ScalerMode.PPO)
            self._base_scenario = replace(self._base_scenario, scaler=scaler_cfg)
        self._scaler_config = scaler_cfg

        self._max_arrivals: int | None = cfg.get('max_arrivals')
        self._max_time_s: float | None = cfg.get('max_time_s')
        self._max_steps: int | None = cfg.get('max_steps')
        if self._max_arrivals is None and self._max_time_s is None and self._max_steps is None:
            self._max_steps = 50
        self._base_seed: int | None = cfg.get('seed')

        # Deriva dimensioni di osservazione e spazio azioni direttamente dallo scenario.
        num_rings = len(region_cfg.infrastructure.rings)
        self._obs_dim = num_rings * FEATURES_PER_RING * self._scaler_config.history_length
        self._action_dim = 1 + 2 * num_rings

        self.observation_space = spaces.Dict({
            'observations': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self._obs_dim,), dtype=np.float32,
            ),
            'action_mask': spaces.Box(
                low=0.0, high=1.0, shape=(self._action_dim,), dtype=np.float32,
            ),
        })
        self.action_space = spaces.Discrete(self._action_dim)

        # Stato dell'episodio corrente.
        self._episode = 0
        self._step_count = 0
        self._sim: Simulation | None = None

    @property
    def simulation(self) -> Simulation | None:
        """Simulazione attiva dell'episodio corrente."""
        return self._sim

    def _make_obs(self, obs_vec: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
        return {
            'observations': np.asarray(obs_vec, dtype=np.float32),
            'action_mask': np.asarray(mask, dtype=np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Avvia una nuova replica della simulazione e restituisce l'osservazione iniziale."""
        super().reset(seed=seed)

        if seed is not None:
            run_seed = int(seed)
        elif self._base_seed is not None:
            run_seed = int(self._base_seed) + self._episode
        else:
            run_seed = self._base_scenario.seed + self._episode
        self._episode += 1
        self._step_count = 0

        # Ricostruisce lo scenario con il seed dell'episodio per avere tracce distinte ma riproducibili.
        from sim.core.simulation import Simulation

        scenario = self._scenario_with_seed(run_seed)
        sim = Simulation(scenario)
        sim._start(attach_scalers=False)

        region = sim.regions[self._region_name]
        scaler = region.scaler
        assert scaler is not None
        scaler.attach(sim._ctx, schedule_steps=False)

        self._sim = sim
        zone_obs = scaler.accounting.zone_observation(sim.now)
        ring_obs, obs_vec = scaler.accounting.observe(sim.now)
        mask = scaler.controller.codec.action_mask(zone_obs)
        info = {
            'sim_time_s': sim.now,
            'seed': run_seed,
            'region': self._region_name,
            'ring_observations': ring_obs,
        }
        return self._make_obs(obs_vec, mask), info

    def step(
        self, action: int | np.integer,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Applica l'azione scelta, avanza la simulazione di Delta e restituisce reward e stato."""
        if self._sim is None:
            raise RuntimeError('Chiamare reset() prima di step()')

        sim = self._sim
        ctx = sim._ctx
        region = sim.regions[self._region_name]
        scaler = region.scaler
        assert scaler is not None

        action_idx = int(action)
        action_kind, ring_id = scaler.controller.codec.decode(action_idx)
        self._step_count += 1

        now = ctx.now
        command = ScalingCommand(
            command_id=f'{self._region_name}-env-{self._step_count}',
            decision_id=self._step_count,
            ring_id=ring_id,
            action=action_kind,
            state_version=scaler.actuator.state_version,
            issued_at_s=now,
        )
        result = scaler.execute_command(command, ctx)

        # Avanza il motore discreto esattamente fino a now + interval_s (o fino a max_arrivals).
        target_time = now + self._scaler_config.interval_s
        if self._max_time_s is not None:
            target_time = min(target_time, self._max_time_s)
        reached = [False]
        ctx.engine.schedule(target_time, _StepBarrierEvent(ctx=ctx, reached=reached))

        calendar_exhausted = False
        while not reached[0]:
            if self._max_arrivals is not None and ctx.metrics.arrivals >= self._max_arrivals:
                break
            if not ctx.engine.step():
                calendar_exhausted = True
                break

        # Chiude l'intervallo [now, ctx.now] e calcola K_t, D_t e reward a due termini.
        interval = scaler.accounting.close_interval(ctx.now, record_step=True)

        mask = scaler.controller.codec.action_mask(scaler.accounting.zone_observation(ctx.now))
        terminated = calendar_exhausted
        truncated = (
            not terminated
            and (
                (self._max_steps is not None and self._step_count >= self._max_steps)
                or (self._max_time_s is not None and ctx.now >= self._max_time_s)
                or (self._max_arrivals is not None and ctx.metrics.arrivals >= self._max_arrivals)
            )
        )

        info: dict[str, Any] = {
            'sim_time_s': ctx.now,
            'arrivals': ctx.metrics.arrivals,
            'action': action_kind.value,
            'ring_id': ring_id,
            'outcome': result.outcome.value,
            'node_name': result.node_name,
            'k_cost': interval.k_cost,
            'd_cost_req_s': interval.d_cost_req_s,
            'energy_kwh': interval.energy_kwh,
            'ring_observations': interval.ring_observations,
        }
        return self._make_obs(interval.feature_vector, mask), float(interval.reward), terminated, truncated, info

    def _scenario_with_seed(self, seed: int) -> Scenario:
        """Restituisce una copia dello scenario con il seed aggiornato e nuovi flussi PRNG."""
        if seed == self._base_scenario.seed:
            return self._base_scenario
        base = self._base_scenario
        functions = [
            replace(fn, sampler=None) for fn in base.function_registry.values()
        ]
        # Ricrea lo scenario preservando esattamente regioni, scaler, load balancer e container.
        new_scenario = object.__new__(Scenario)
        object.__setattr__(new_scenario, 'seed', seed)
        object.__setattr__(new_scenario, 'load_balancer', base.load_balancer)
        object.__setattr__(new_scenario, 'load_balancer_replicas', base.load_balancer_replicas)
        object.__setattr__(new_scenario, 'scaler', base.scaler)
        object.__setattr__(new_scenario, 'scheduler', base.scheduler)
        object.__setattr__(new_scenario, 'containers', base.containers)
        object.__setattr__(new_scenario, 'primary_region', base.primary_region)
        object.__setattr__(new_scenario, 'regions', base.regions)
        object.__setattr__(new_scenario, 'invocation', base.invocation)
        registry = new_scenario._register_functions(functions)
        object.__setattr__(new_scenario, 'function_registry', dict(registry))
        return new_scenario
