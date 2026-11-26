"""Test unitari e di integrazione dello scaler regionale RL (attuatore, contabilità, Gymnasium, Ray RLlib PPO).

Verifica il contratto del controllo di autoscaling regionale:
- configurazione selettiva per regione;
- spazio azioni, mascheramento vincoli (`min_nodes`/`max_nodes`/quota) e drain deterministico;
- contabilità energetica continua $K_t$, attese $D_t$ e reward a due termini;
- conformità `gymnasium.utils.env_checker.check_env`, checkpoint RLlib PPO e
  intercambiabilità di `ZoneController` con un `ScalingBackend` esterno (es. gRPC verso Infrascaler).
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from gymnasium.utils.env_checker import check_env
from scaler import (
    CommandOutcome, FEATURES_PER_RING, PPOConfig, RLlibPPOPolicy,
    RewardConfig, RingObservation, ScalingActionKind, ScalingBackend, ScalingCommand,
    ScalingResult, ZoneController, ZoneObservation, load_rllib_module,
)
from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario
from simulator.model.execution.request import Request
from simulator.model.scaling import PPOScaler
from simulator.observer.notifications import NotificationType
from simulator.random.streams import RandomStreams, StreamKind
from simulator.utils.report import format_metrics
from trainer import ServerledgeScalingEnv, train_rllib_ppo


def _two_ring_scenario(
    *,
    scaler: dict | str | None = None,
    x86_scaling: dict | None = None,
    arm_scaling: dict | None = None,
    x86_nodes: list[dict] | None = None,
    arm_nodes: list[dict] | None = None,
    arrival_rate: float = 2.0,
    service_rate: float = 2.0,
    seed: int = 42,
) -> Scenario:
    """Costruisce uno scenario con due ring omogenei (x86 e arm64) in una singola regione."""
    ring_x86: dict = {
        'id': 'edge-x86',
        'arch': 'amd64',
        'memory_mb': 512,
        'cpus': 2.0,
        'nodes': [{'name': 'x86-1'}] if x86_nodes is None else x86_nodes,
    }
    if x86_scaling is not None:
        ring_x86['scaling'] = x86_scaling

    ring_arm: dict = {
        'id': 'edge-arm',
        'arch': 'arm64',
        'memory_mb': 512,
        'cpus': 2.0,
        'nodes': [{'name': 'arm-1'}] if arm_nodes is None else arm_nodes,
    }
    if arm_scaling is not None:
        ring_arm['scaling'] = arm_scaling

    return Scenario(
        functions=[
            {
                'name': 'fn-any',
                'runtime': 'python314',
                'memory_mb': 128,
                'cpu_demand': 1.0,
                'supported_archs': ['amd64', 'arm64'],
                'arrival_rate': arrival_rate,
                'service_rate': service_rate,
                'first_arrival_time': 0.5,
                'arrival_distribution': 'deterministic',
                'service_distribution': 'deterministic',
            },
        ],
        seed=seed,
        load_balancer='RoundRobin',
        scaler=scaler,
        containers={'idle_timeout': 10.0, 'cleanup_interval': 5.0},
        infrastructure={'rings': [ring_x86, ring_arm]},
    )


class TestActuatorAndLifecycle(unittest.TestCase):
    """Verifica configurazione per regione, spazio azioni, maschere, quote e drain dei nodi."""

    def test_per_region_scaler_override(self) -> None:
        """Verifica l'attivazione selettiva dello scaler PPO su una singola regione dello scenario.

        - Configura due regioni simulate (`edge` con `scaler={'mode': 'PPO'}` e `backup` senza scaler).
        - Verifica che `Simulation` istanzi `PPOScaler` solo su `edge` e lasci `backup.scaler` a `None`.
        """
        scenario = Scenario(
            functions=[{
                'name': 'f1',
                'runtime': 'python314',
                'memory_mb': 128,
                'cpu_demand': 1.0,
                'arrival_rate': 1.0,
                'service_rate': 2.0,
                'first_arrival_time': 0.5,
            }],
            primary_region='edge',
            regions={
                'edge': {
                    'scaler': {'mode': 'PPO', 'interval_s': 2.0},
                    'infrastructure': {
                        'rings': [{
                            'id': 'edge-x86',
                            'arch': 'amd64',
                            'memory_mb': 512,
                            'cpus': 2.0,
                            'nodes': [{'name': 'n1'}],
                            'scaling': {'min_nodes': 1, 'max_nodes': 3},
                        }],
                    },
                },
                'backup': {
                    'infrastructure': {
                        'rings': [{
                            'id': 'backup-x86',
                            'arch': 'amd64',
                            'memory_mb': 512,
                            'cpus': 2.0,
                            'nodes': [{'name': 'b1'}],
                        }],
                    },
                },
            },
        )
        sim = Simulation(scenario)
        self.assertIsInstance(sim.regions['edge'].scaler, PPOScaler)
        self.assertIsNone(sim.regions['backup'].scaler)

    def test_action_space_and_masks_for_two_rings(self) -> None:
        """Verifica codifica azioni, mascheramento limiti `min_nodes`/`max_nodes` e rifiuto comandi oltre quota.

        - Con 2 ring (`edge-x86` [1..2] con 1 nodo, `edge-arm` [0..1] con 1 nodo), `action_dim == 5`
          (`HOLD`, `SCALE_OUT/IN` per ciascun ring) e la maschera iniziale è `[T, T, F, F, T]`.
        - Un `SCALE_OUT` su `edge-x86` porta `provisioning_count=1`, esaurisce `remaining_quota=0`
          e disabilita subito l'azione `SCALE_OUT` nella maschera; un secondo `SCALE_OUT` viene rifiutato.
        """
        scenario = _two_ring_scenario(
            scaler={'mode': 'PPO', 'interval_s': 2.0},
            x86_scaling={'min_nodes': 1, 'max_nodes': 2, 'boot_time_s': 3.0},
            arm_scaling={'min_nodes': 0, 'max_nodes': 1, 'boot_time_s': 3.0},
        )
        sim = Simulation(scenario)
        scaler = sim.regions['default'].scaler
        assert isinstance(scaler, PPOScaler)
        act = scaler.actuator
        codec = scaler.controller.codec

        self.assertEqual(codec.action_dim, 5)
        self.assertEqual(codec.decode(0), (ScalingActionKind.HOLD, None))
        self.assertEqual(codec.decode(1), (ScalingActionKind.SCALE_OUT, 'edge-x86'))
        self.assertEqual(codec.decode(2), (ScalingActionKind.SCALE_IN, 'edge-x86'))
        self.assertEqual(codec.decode(3), (ScalingActionKind.SCALE_OUT, 'edge-arm'))
        self.assertEqual(codec.decode(4), (ScalingActionKind.SCALE_IN, 'edge-arm'))

        mask = codec.action_mask(scaler.accounting.zone_observation(0.0))
        self.assertEqual(mask.tolist(), [True, True, False, False, True])

        cmd_out = ScalingCommand('c1', 1, 'edge-x86', ScalingActionKind.SCALE_OUT, act.state_version, 0.0)
        res = act.execute(cmd_out, now=0.0)
        self.assertEqual(res.outcome, CommandOutcome.IN_PROGRESS)
        self.assertEqual(act.rings['edge-x86'].provisioning_count, 1)
        self.assertEqual(act.rings['edge-x86'].remaining_quota, 0)
        self.assertFalse(codec.action_mask(scaler.accounting.zone_observation(0.0))[1])

        cmd_dup = ScalingCommand('c2', 2, 'edge-x86', ScalingActionKind.SCALE_OUT, act.state_version, 0.0)
        res_dup = act.execute(cmd_dup, now=0.0)
        self.assertEqual(res_dup.outcome, CommandOutcome.REJECTED)
        self.assertEqual(act.rejected_commands, 1)

    def test_deterministic_scale_in_and_drain_completion(self) -> None:
        """Verifica la scelta deterministica del candidato scale-in (nodo meno carico) e il drain graceful.

        - A parità di carico (`0`), il candidato deterministico è `x86-a`; dopo aver sottomesso una
          richiesta su `x86-a`, il candidato diventa il nodo libero `x86-b`.
        - Il primo `SCALE_IN` rimuove immediatamente `x86-b` (`COMPLETED`, `draining_count=0`).
        - Il secondo `SCALE_IN` su `x86-a` (occupato) lo esclude subito dal ring in stato `IN_PROGRESS`
          (`draining_count=1`) e completa lo spegnimento al termine della richiesta in volo.
        """
        scenario = _two_ring_scenario(
            scaler={'mode': 'PPO', 'interval_s': 10.0},
            x86_nodes=[{'name': 'x86-a'}, {'name': 'x86-b'}],
            x86_scaling={'min_nodes': 0, 'max_nodes': 3, 'shutdown_time_s': 0.0},
            arm_nodes=[],
            arm_scaling={'min_nodes': 0, 'max_nodes': 1},
            arrival_rate=1.0,
            service_rate=0.5,
        )
        sim = Simulation(scenario)
        region = sim.regions['default']
        scaler = region.scaler
        assert isinstance(scaler, PPOScaler)
        sim._ctx.observer.subscribe(scaler.accounting.on_notification, propagate_errors=True)

        ring = region.infrastructure.rings['edge-x86']  # type: ignore[union-attr]
        node_a = next(n for n in ring if n.key == 'x86-a')
        node_b = next(n for n in ring if n.key == 'x86-b')

        ring_act = scaler.actuator.rings['edge-x86']
        self.assertIs(ring_act.select_scale_in_candidate(), node_a)

        fn = scenario.function_registry['fn-any']
        req = Request(fn, service_duration=2.0, arrival_time=0.0)
        sim._build_request_bindings().bind(req)
        region.submit(req, now=0.0, node=node_a)
        self.assertIs(ring_act.select_scale_in_candidate(), node_b)

        cmd_b = ScalingCommand('c1', 1, 'edge-x86', ScalingActionKind.SCALE_IN, 0, 0.0)
        res_b = scaler.actuator.execute(cmd_b, now=0.0)
        self.assertEqual(res_b.node_name, 'x86-b')
        self.assertEqual(res_b.outcome, CommandOutcome.COMPLETED)
        self.assertEqual(ring_act.draining_count, 0)
        self.assertEqual(ring_act.scale_in_completed, 1)

        cmd_a = ScalingCommand('c2', 2, 'edge-x86', ScalingActionKind.SCALE_IN, 1, 0.0)
        res_a = scaler.actuator.execute(cmd_a, now=0.0)
        self.assertEqual(res_a.node_name, 'x86-a')
        self.assertEqual(res_a.outcome, CommandOutcome.IN_PROGRESS)
        self.assertNotIn(node_a, ring)
        self.assertEqual(ring_act.draining_count, 1)

        while sim._ctx.engine.step():
            pass
        self.assertEqual(ring_act.draining_count, 0)
        self.assertEqual(ring_act.scale_in_completed, 2)
        self.assertEqual(ring_act.remaining_quota, 3)


class TestAccountingAndReward(unittest.TestCase):
    """Verifica contabilità continua di energia K_t, attese/penalità D_t e reward a due termini."""

    def test_idle_energy_and_reward_formula(self) -> None:
        """Verifica il calcolo esatto dell'energia idle in kWh e del reward normalizzato senza carico.

        - Un nodo da `100 W` idle per `36 s` consuma esattamente `100 * 36 / 3_600_000 = 0.001 kWh`.
        - Con `w_k=2.0`, `k_ref=0.001` e `d_cost=0.0`, il reward deve essere esattamente `-2.0`.
        """
        scenario = _two_ring_scenario(
            scaler={
                'mode': 'PPO',
                'interval_s': 36.0,
                'reward': {
                    'w_k': 2.0,
                    'w_t': 3.0,
                    'k_ref': 0.001,
                    'd_ref': 10.0,
                },
            },
            x86_nodes=[{'name': 'x86-1'}],
            arm_nodes=[],
            x86_scaling={'min_nodes': 1, 'max_nodes': 1, 'idle_power_w': 100.0, 'max_power_w': 200.0},
            arm_scaling={'min_nodes': 0, 'max_nodes': 1, 'idle_power_w': 50.0, 'max_power_w': 100.0},
        )
        sim = Simulation(scenario)
        scaler = sim.regions['default'].scaler
        assert isinstance(scaler, PPOScaler)

        snap = scaler.accounting.close_interval(36.0)
        self.assertAlmostEqual(snap.energy_kwh, 0.001, places=12)
        self.assertAlmostEqual(snap.k_cost, 0.001, places=12)
        self.assertAlmostEqual(snap.d_cost_req_s, 0.0, places=12)
        self.assertAlmostEqual(snap.reward, -2.0 * (0.001 / 0.001), places=12)

    def test_all_off_rejection_penalty_accounting(self) -> None:
        """Verifica la contabilizzazione delle penalità di rifiuto quando tutti i nodi sono spenti.

        - Con 0 nodi attivi e `timeout_penalty_s=10.0`, 5 arrivi rifiutati generano `0.0 kWh`
          di energia, `50.0 req·s` di penalità in `D_t` e reward totale `-5.0` (`d_ref=10.0`).
        """
        scenario = _two_ring_scenario(
            scaler={
                'mode': 'PPO',
                'interval_s': 10.0,
                'reward': {
                    'w_k': 1.0,
                    'w_t': 1.0,
                    'k_ref': 1.0,
                    'd_ref': 10.0,
                    'timeout_penalty_s': 10.0,
                },
            },
            x86_nodes=[],
            arm_nodes=[],
            x86_scaling={'min_nodes': 0, 'max_nodes': 1},
            arm_scaling={'min_nodes': 0, 'max_nodes': 1},
            arrival_rate=1.0,
        )
        sim = Simulation(scenario)
        scaler = sim.regions['default'].scaler
        assert isinstance(scaler, PPOScaler)
        for ring_act in scaler.actuator.rings.values():
            ring_act._max_nodes = 0

        stats = sim.run(arrivals=5)
        costs = stats.metrics['regions']['default']['scaler']['costs']
        self.assertAlmostEqual(costs['total_energy_kwh'], 0.0, places=12)
        self.assertEqual(stats.metrics['total']['rejected'], 5)
        self.assertAlmostEqual(costs['total_rejection_penalty_req_s'], 50.0, places=9)
        self.assertAlmostEqual(costs['total_d_cost_req_s'], 50.0, places=9)
        self.assertAlmostEqual(costs['total_reward'], -5.0, places=9)


class TestGymnasiumEnvAndRLlibPPO(unittest.TestCase):
    """Verifica conformità Gymnasium, inferenza mascherata, training Ray RLlib PPO e backend esterno."""

    def test_gymnasium_env_checker_and_step_contract(self) -> None:
        """Verifica che `ServerledgeScalingEnv` superi `gymnasium.utils.env_checker.check_env` e rispetti `max_steps`.

        - Controlla forma di `observations` (con `history_length=2`) e `action_mask` (5 azioni).
        - Esegue un episodio completo fino a `truncated=True` verificando `steps == 5`,
          `reward <= 0` ed `energy_kwh >= 0`.
        """
        scenario = _two_ring_scenario(
            scaler={'mode': 'PPO', 'interval_s': 2.0, 'history_length': 2},
            x86_scaling={'min_nodes': 1, 'max_nodes': 3, 'boot_time_s': 1.0},
            arm_scaling={'min_nodes': 0, 'max_nodes': 3, 'boot_time_s': 1.0},
        )
        env = ServerledgeScalingEnv({'scenario': scenario, 'max_steps': 5})
        check_env(env, skip_render_check=True)

        obs, info = env.reset(seed=100)
        self.assertEqual(obs['observations'].shape, (2 * FEATURES_PER_RING * 2,))
        self.assertEqual(obs['action_mask'].shape, (5,))
        self.assertEqual(info['sim_time_s'], 0.0)

        truncated = False
        steps = 0
        while not truncated:
            valid = np.flatnonzero(obs['action_mask'] > 0.5)
            obs, reward, terminated, truncated, step_info = env.step(int(valid[0]))
            self.assertFalse(terminated)
            self.assertLessEqual(reward, 0.0)
            self.assertGreaterEqual(step_info['energy_kwh'], 0.0)
            steps += 1
        self.assertEqual(steps, 5)

    def test_rllib_policy_masking_and_checkpoint_roundtrip(self) -> None:
        """Verifica che `RLlibPPOPolicy` rispetti rigorosamente l'action mask e preservi l'inferenza dopo save/load.

        - In modalità stocastica estrae solo le azioni consentite dalla maschera (`0` o `2`).
        - Dopo `save()` + `load_rllib_module()`, la selezione deterministica produce la stessa azione.
        """
        rng = RandomStreams(7).stream(StreamKind.SCALING, 'test-rllib')
        cfg = PPOConfig(hidden_sizes=(16, 16), deterministic=True)
        policy = RLlibPPOPolicy(obs_dim=6, action_dim=5, config=cfg, seed=7)

        mask = np.array([True, False, True, False, False], dtype=bool)
        obs = np.array([0.1, -0.3, 0.5, 0.2, 0.0, 1.0], dtype=np.float32)
        for _ in range(10):
            action = policy.select_action(obs, mask, rng=rng, deterministic=False)
            self.assertIn(action, (0, 2))

        expected = policy.select_action(obs, mask, rng=rng, deterministic=True)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = policy.save(Path(tmp) / 'rl_module')
            reloaded_module = load_rllib_module(ckpt)
            reloaded = RLlibPPOPolicy(obs_dim=6, action_dim=5, config=cfg, module=reloaded_module)
            self.assertEqual(reloaded.select_action(obs, mask, rng=rng, deterministic=True), expected)

    def test_rllib_ppo_train_and_load_in_simulation(self) -> None:
        """Verifica il ciclo end-to-end: training `train_rllib_ppo`, salvataggio checkpoint e valutazione riproducibile.

        - Addestra 1 iterazione PPO su uno scenario a 2 ring e salva il checkpoint `RLModule`.
        - Ricarica lo stesso checkpoint in due repliche `Simulation` con identico seed e verifica
          identità del reward totale, coerenza delle notifiche `SCALER_DECISION` e presenza della
          sezione `Autoscaling regionale` nel report.
        """
        scaler_cfg = {
            'mode': 'PPO',
            'interval_s': 2.0,
            'reward': {
                'w_k': 1.0,
                'w_t': 1.0,
                'k_ref': 0.01,
                'd_ref': 5.0,
                'timeout_penalty_s': 10.0,
            },
            'ppo': {
                'hidden_sizes': [16, 16],
                'rollout_steps': 12,
                'ppo_epochs': 2,
                'minibatch_size': 6,
                'deterministic': True,
            },
        }
        scenario = _two_ring_scenario(
            scaler=scaler_cfg,
            x86_scaling={'min_nodes': 1, 'max_nodes': 3, 'boot_time_s': 1.0, 'idle_power_w': 60.0, 'max_power_w': 160.0},
            arm_scaling={'min_nodes': 0, 'max_nodes': 3, 'boot_time_s': 1.0, 'idle_power_w': 30.0, 'max_power_w': 80.0},
            arrival_rate=2.0,
            service_rate=1.0,
            seed=123,
        )

        with tempfile.TemporaryDirectory() as tmp:
            result = train_rllib_ppo(
                scenario,
                num_iterations=1,
                max_steps=6,
                num_env_runners=0,
                checkpoint_dir=Path(tmp) / 'ppo_ckpt',
                seed=123,
            )
            self.assertEqual(len(result['history']), 1)
            self.assertIsNotNone(result['checkpoint_path'])

            eval_scaler = {
                **scaler_cfg,
                'ppo': {**scaler_cfg['ppo'], 'checkpoint_path': result['checkpoint_path']},
            }
            s1 = _two_ring_scenario(
                scaler=eval_scaler,
                x86_scaling={'min_nodes': 1, 'max_nodes': 3, 'boot_time_s': 1.0, 'idle_power_w': 60.0, 'max_power_w': 160.0},
                arm_scaling={'min_nodes': 0, 'max_nodes': 3, 'boot_time_s': 1.0, 'idle_power_w': 30.0, 'max_power_w': 80.0},
                arrival_rate=2.0,
                service_rate=1.0,
                seed=123,
            )
            s2 = _two_ring_scenario(
                scaler=eval_scaler,
                x86_scaling={'min_nodes': 1, 'max_nodes': 3, 'boot_time_s': 1.0, 'idle_power_w': 60.0, 'max_power_w': 160.0},
                arm_scaling={'min_nodes': 0, 'max_nodes': 3, 'boot_time_s': 1.0, 'idle_power_w': 30.0, 'max_power_w': 80.0},
                arrival_rate=2.0,
                service_rate=1.0,
                seed=123,
            )

            decisions_seen: list[dict] = []
            sim1 = Simulation(s1)
            sim1._ctx.observer.subscribe(
                lambda n: decisions_seen.append(n.to_dict()) if n.type is NotificationType.SCALER_DECISION else None
            )
            stats1 = sim1.run(arrivals=30)
            stats2 = Simulation(s2).run(arrivals=30)

            m1 = stats1.metrics['regions']['default']['scaler']
            m2 = stats2.metrics['regions']['default']['scaler']

            self.assertEqual(m1['algorithm']['backend'], 'ray.rllib')
            self.assertEqual(len(decisions_seen), m1['decisions'])
            self.assertGreater(m1['costs']['total_energy_kwh'], 0.0)
            self.assertGreater(m1['costs']['total_d_cost_req_s'], 0.0)
            self.assertAlmostEqual(m1['costs']['total_reward'], m2['costs']['total_reward'], places=12)

            report = format_metrics(stats1.metrics)
            self.assertIn('Autoscaling regionale', report)
            self.assertIn('ray.rllib', report)

    def test_zone_controller_interchangeable_with_external_backend(self) -> None:
        """Verifica che `ZoneController` + `RLlibPPOPolicy` operino su un `ScalingBackend` esterno (es. gRPC Go).

        - Collega il controller a un backend implementante il solo protocollo `ScalingBackend`
          senza alcuna dipendenza dal simulatore.
        - Verifica che `control_step()` esegua un comando valido rispetto ai vincoli di maschera
          e calcoli il reward esatto (`-2.0`).
        """
        class _MockGrpcBackend(ScalingBackend):
            def __init__(self) -> None:
                self.executed: list[ScalingCommand] = []

            @property
            def region_id(self) -> str:
                return 'edge-real'

            @property
            def ring_ids(self) -> tuple[str, ...]:
                return ('ring-x86', 'ring-arm')

            def observe(self) -> ZoneObservation:
                return ZoneObservation(
                    region_id=self.region_id,
                    timestamp_s=10.0,
                    state_version=3,
                    rings=(
                        RingObservation(
                            ring_id='ring-x86', recent_arrivals=12, backlog_count=2,
                            backlog_max_age_s=0.8, cpu_utilization=0.75, memory_utilization=0.5,
                            ready_nodes=1, provisioning_nodes=0, draining_nodes=0,
                            est_remaining_boot_s=0.0, remaining_quota=2, min_nodes=1, max_nodes=3,
                            boot_time_s=5.0, interval_energy_kwh=0.002, interval_delay_req_s=1.5,
                            interval_rejections=0, can_scale_out=True, can_scale_in=False,
                        ),
                        RingObservation(
                            ring_id='ring-arm', recent_arrivals=4, backlog_count=0,
                            backlog_max_age_s=0.0, cpu_utilization=0.1, memory_utilization=0.1,
                            ready_nodes=1, provisioning_nodes=0, draining_nodes=0,
                            est_remaining_boot_s=0.0, remaining_quota=0, min_nodes=0, max_nodes=1,
                            boot_time_s=5.0, interval_energy_kwh=0.001, interval_delay_req_s=0.2,
                            interval_rejections=0, can_scale_out=False, can_scale_in=True,
                        ),
                    ),
                    interval_energy_kwh=0.003,
                    interval_wait_req_s=1.7,
                )

            def execute(self, command: ScalingCommand) -> ScalingResult:
                self.executed.append(command)
                return ScalingResult(command=command, outcome=CommandOutcome.ACCEPTED)

        backend = _MockGrpcBackend()
        policy = RLlibPPOPolicy(
            obs_dim=2 * FEATURES_PER_RING,
            action_dim=5,
            config=PPOConfig(hidden_sizes=(16,), deterministic=True),
            seed=9,
        )
        controller = ZoneController(
            region_id=backend.region_id,
            ring_ids=backend.ring_ids,
            policy=policy,
            reward_config=RewardConfig(w_k=1.0, w_t=1.0, k_ref=0.003, d_ref=1.7),
            interval_s=5.0,
            rng=RandomStreams(9).stream(StreamKind.SCALING, 'mock-grpc'),
        )

        _, cmd, res, breakdown = controller.control_step(backend)
        self.assertEqual(len(backend.executed), 1)
        self.assertEqual(res.outcome, CommandOutcome.ACCEPTED)
        self.assertNotEqual((cmd.action, cmd.ring_id), (ScalingActionKind.SCALE_IN, 'ring-x86'))
        self.assertNotEqual((cmd.action, cmd.ring_id), (ScalingActionKind.SCALE_OUT, 'ring-arm'))
        self.assertAlmostEqual(breakdown.reward, -2.0, places=12)


if __name__ == '__main__':
    unittest.main()
