"""Verifiche degli arrivi e del ciclo di vita dei container.

Riferimenti Infrascaler:
- internal/scheduling/policy_default.go: FIFO opzionale e ripartenza al completamento.
- internal/scheduling/queue.go: ordine di accodamento e capacità.
- internal/container/runtimes.go: immagini Docker dei runtime integrati.
- internal/container/container.go: scelta dell'immagine e copia del codice.
- internal/node/pool.go: riuso del container e scadenza delle istanze idle.
- internal/node/janitor_handler.go: pulizia periodica dei container scaduti.
"""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sim.io.scenario import Scenario
from sim.model.function import BuiltinRuntime, CustomRuntime, Function, FunctionProfile
from sim.core.metrics import SimulationMetrics
from sim.core.simulation import Simulation
from sim.model.container import ContainerState
from sim.model.request import RequestStatus
from sim.model.infrastructure import Infrastructure
from sim.observer import Notification, NotificationType, SimulationError, observer
from sim.__main__ import main
from sim.utils.welford import TimedWelford, Welford


def profiled_function(name, arrival_rate, service_rate, first_arrival_time=0):
    return Function(name, 128, 1.0, BuiltinRuntime('python314'),
                    profile=FunctionProfile(arrival_rate, service_rate, first_arrival_time))


EDGE = {'rings': [{
    'id': 'edge', 'arch': 'amd64', 'memory_mb': 128, 'cpus': 1,
    'nodes': [{'name': 'n1'}],
}]}


def record_submissions(simulation):
    requests = []
    simulation._ctx.infrastructure.on_request_submitted = lambda infra, request, node: requests.append(request)
    return requests


class SimulationTest(unittest.TestCase):
    def advance_to(self, engine, time):
        """Esegue fino a un istante in cui è previsto almeno un evento."""
        while engine.now < time:
            self.assertTrue(engine.step())
        self.assertEqual(engine.now, time)

    def test_request_notifications_correlate_arrivals_with_outcomes(self):
        """Il primo arrivo completa e il bilanciatore rifiuta il secondo per saturazione."""
        function = Function('hello', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(
            arrival_rate=1, service_rate=0.4, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))
        scenario = Scenario((function,), infrastructure=EDGE)
        notifications: list[Notification] = []
        self.addCleanup(observer.subscribe(notifications.append))

        Simulation(scenario).run(arrivals=4)

        payloads = [json.loads(json.dumps(event.to_dict())) for event in notifications
                    if event.type in (NotificationType.REQUEST_ARRIVED, NotificationType.REQUEST_REJECTED,
                                      NotificationType.REQUEST_COMPLETED)]
        first, second, rejected = [payload['data'] for payload in payloads[:3]]
        self.assertEqual([payload['type'] for payload in payloads[:3]],
                         ['request.arrived', 'request.arrived', 'request.rejected'])
        self.assertEqual(rejected['request_id'], second['request_id'])
        self.assertEqual(rejected['reason'], 'out_of_resources')
        self.assertIsNone(rejected['node_id'])
        completed, = [payload['data'] for payload in payloads if payload['type'] == 'request.completed']
        self.assertEqual(completed['request_id'], first['request_id'])
        self.assertEqual(completed['response_time_s'], 2.5)

    def test_final_statistics_can_be_emitted_after_run(self):
        """Il chiamante pubblica lo snapshot serializzabile dopo il ritorno di run."""
        function = profiled_function('hello', 1, 1, first_arrival_time=2)
        simulation = Simulation(Scenario((function,)))
        notifications: list[Notification] = []
        self.addCleanup(observer.subscribe(notifications.append))

        statistics = simulation.run(arrivals=1)
        self.assertFalse(any(event.type is NotificationType.FINAL_STATISTICS for event in notifications))
        observer.notify(statistics)

        event, = [event for event in notifications if event.type is NotificationType.FINAL_STATISTICS]
        payload = json.loads(json.dumps(event.to_dict()))
        self.assertEqual(payload['type'], NotificationType.FINAL_STATISTICS)
        self.assertEqual(payload['data']['outcome'], 'completed')
        self.assertEqual(payload['data']['metrics'], simulation.metrics_snapshot())
        self.assertEqual(payload['sim_time_s'], 2)

    def test_cli_prints_one_structured_report_and_keeps_logs_separate(self):
        root = Path(__file__).resolve().parents[1]
        scenario = 'examples/scenarios/regional-offloading.toml'
        command = [sys.executable, '-m', 'sim', scenario, '--arrivals', '5']
        for level in ('WARNING', 'INFO'):
            with self.subTest(level=level):
                result = subprocess.run(
                    [*command, '--log-level', level], cwd=root,
                    capture_output=True, text=True, check=True, timeout=10,
                )
                self.assertEqual(result.stdout.count('Riepilogo simulazione'), 1)
                self.assertIn(scenario, result.stdout)
                self.assertRegex(result.stdout, r'(?m)^    In transito +1$')
                self.assertRegex(result.stdout, r'(?m)^  Risposta\s+0\.625 s$')
                self.assertIn('Invocazioni per funzione', result.stdout)
                self.assertIn('edge (primaria)', result.stdout)
                self.assertIn('Offloading interregionale', result.stdout)
                self.assertRegex(result.stdout, r'(?m)^  edge\s+cloud\s+5$')
                self.assertNotIn('request.arrived', result.stdout)
                self.assertNotIn('simulation.final_statistics', result.stderr)
                self.assertEqual('request.arrived' in result.stderr, level == 'INFO')

    def test_cli_reports_simulation_failure(self):
        failure = ValueError('inoltro fallito')
        scenario = Scenario((profiled_function('hello', 1, 1),))
        notifications = []
        self.addCleanup(observer.subscribe(notifications.append))

        with (
            patch('sys.argv', ['sim', 'scenario.toml', '--arrivals', '1']),
            patch.object(Scenario, 'load', return_value=scenario),
            patch.object(Infrastructure, 'submit', side_effect=failure),
            self.assertLogs('sim.notifications', level='ERROR') as logs,
            self.assertRaises(SystemExit) as raised,
        ):
            main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIs(raised.exception.__cause__, failure)
        event, = [event for event in notifications if isinstance(event, SimulationError)]
        self.assertEqual(event.to_dict()['data'], {
            'failed_event': 'Simulation Loop',
            'error_type': 'ValueError',
            'message': 'inoltro fallito',
        })
        self.assertIn('ValueError: inoltro fallito', logs.output[0])

    def test_missing_destination_is_a_rejection_without_a_node(self):
        function = profiled_function('hello', 1, 1)
        simulation = Simulation(Scenario((function,)))
        notifications = []
        self.addCleanup(observer.subscribe(notifications.append))

        statistics = simulation.run(arrivals=1)

        arrival, rejected = [notification.to_dict() for notification in notifications]
        self.assertEqual(rejected['type'], 'request.rejected')
        self.assertEqual(rejected['data']['request_id'], arrival['data']['request_id'])
        self.assertEqual(rejected['data']['reason'], 'no_destination')
        self.assertIsNone(rejected['data']['node_id'])
        totals = statistics.metrics['total']
        self.assertEqual((totals['arrivals'], totals['rejected'], totals['pending']), (1, 1, 0))

    def test_same_seed_reproduces_arrival_and_service_samples(self):
        function = profiled_function('hello', arrival_rate=3, service_rate=4)
        scenario = Scenario(functions=(function,), seed=42)
        observations = []
        for _ in range(2):
            simulation = Simulation(scenario)
            requests = record_submissions(simulation)
            simulation.run(arrivals=5)
            self.assertEqual(len(requests), 5)
            observations.append([(request.arrival_time, request.service_duration) for request in requests])
        self.assertEqual(observations[0], observations[1])

    def test_metrics_subscriber_failure_stops_the_replica(self):
        failure = RuntimeError('metriche non aggiornate')
        scenario = Scenario((profiled_function('hello', 1, 1),))
        with patch.object(SimulationMetrics, '_record_invocation', side_effect=failure):
            with self.assertRaises(RuntimeError) as raised:
                Simulation(scenario).run(arrivals=1)
        self.assertIs(raised.exception, failure)

    def test_example_scenario_formats_define_the_same_workload(self):
        examples = Path(__file__).resolve().parents[1] / 'examples/scenarios'
        scenario = Scenario.load(examples / 'startup.toml')
        self.assertEqual(scenario.load_balancer, 'Random')
        for suffix in ('json', 'yaml'):
            with self.subTest(format=suffix):
                self.assertEqual(scenario, Scenario.load(examples / f'startup.{suffix}'))

    def test_scenario_load_balancer_mode_is_validated_and_used_at_startup(self):
        source = Path(__file__).resolve().parents[1] / 'examples/scenarios/startup.json'
        definition = json.loads(source.read_text(encoding='utf-8'))
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scenario.json'
            for mode in ('MAB', 'RoundRobin', 'ArchitectureUnaware'):
                with self.subTest(mode=mode):
                    definition['load_balancer'] = mode
                    path.write_text(json.dumps(definition), encoding='utf-8')
                    scenario = Scenario.load(path)
                    self.assertEqual(scenario.load_balancer, mode)
                    with self.assertRaisesRegex(NotImplementedError, mode):
                        Simulation(scenario)

            definition['load_balancer'] = 'Other'
            path.write_text(json.dumps(definition), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'load_balancer'):
                Scenario.load(path)

    def test_simulation_seed_controls_ring_selection(self):
        infrastructure = {'rings': [
            {'id': name, 'arch': 'amd64', 'memory_mb': 512, 'cpus': 2,
             'nodes': [{'name': name}]}
            for name in ('first', 'second')
        ]}
        fun = Function('hello', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(
            arrival_rate=1, service_rate=4, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))

        def destinations(seed):
            simulation = Simulation(Scenario(
                (fun,), seed=seed, infrastructure=infrastructure, load_balancer='Random',
            ))
            requests = record_submissions(simulation)
            simulation.run(arrivals=20)
            return [request.node.id.key for request in requests]

        first = destinations(42)
        self.assertEqual(set(first), {'first', 'second'})
        self.assertEqual(first, destinations(42))
        self.assertNotEqual(first, destinations(43))

    def test_arrival_rate_scales_interarrival_times(self):
        function = profiled_function('hello', arrival_rate=2, service_rate=4, first_arrival_time=1)
        faster = replace(function, profile=replace(function.profile, arrival_rate=4))
        observations = []
        for profile in (function, faster):
            simulation = Simulation(Scenario((profile,), seed=42))
            requests = record_submissions(simulation)
            simulation.run(arrivals=5)
            observations.append([request.arrival_time for request in requests])
        baseline_times, faster_times = observations
        self.assertEqual(baseline_times[0], 1)
        self.assertEqual(faster_times[0], 1)
        for baseline, faster in zip(baseline_times[1:], faster_times[1:], strict=True):
            self.assertAlmostEqual(baseline - 1, 2 * (faster - 1))

    def test_function_profile_rejects_invalid_rates(self):
        for field in ('arrival_rate', 'service_rate'):
            for value in (0, -1, float('inf'), True):
                with self.subTest(field=field, value=value):
                    rates = {'arrival_rate': 2, 'service_rate': 4, 'first_arrival_time': 0, field: value}
                    with self.assertRaises(ValueError):
                        FunctionProfile(**rates)

    def test_service_rate_scales_service_duration_without_shifting_arrivals(self):
        observations = []
        for rate in (4, 8):
            simulation = Simulation(Scenario((profiled_function('hello', 2, rate),)))
            requests = record_submissions(simulation)
            simulation.run(arrivals=5)
            observations.append(requests)

        for baseline, faster in zip(*observations, strict=True):
            self.assertEqual(baseline.arrival_time, faster.arrival_time)
            self.assertGreater(baseline.service_duration, 0)
            self.assertAlmostEqual(baseline.service_duration, 2 * faster.service_duration)

    def test_functions_keep_independent_arrival_and_service_streams(self):
        hello = profiled_function('hello', 2, 4, first_arrival_time=1)
        resize = profiled_function('resize', 3, 8, first_arrival_time=0.5)
        single = Simulation(Scenario((hello,)))
        single_requests = record_submissions(single)
        single.run(arrivals=5)

        multiple = Simulation(Scenario((resize, hello)))
        multiple_requests = record_submissions(multiple)
        multiple.run(arrivals=40)

        self.assertEqual(multiple_requests[0].function.name, 'resize')
        hello_requests = [request for request in multiple_requests if request.function.name == 'hello']
        self.assertEqual(
            [(request.arrival_time, request.service_duration) for request in hello_requests[:5]],
            [(request.arrival_time, request.service_duration) for request in single_requests],
        )

    def test_scenario_requires_at_least_one_profile_with_unique_name(self):
        function = profiled_function('hello', 2, 4)
        unprofiled = Function('bare', 128, 1.0, BuiltinRuntime('python314'))
        for functions in ((), (function, function), (unprofiled,)):
            with self.assertRaises(ValueError):
                Scenario(functions)

    def test_legacy_single_function_file_loads_as_one_profile(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.toml'
            path.write_text('[function]\nname = "hello"\n'
                            'runtime = "python314"\nmemory_mb = 128\ncpu_demand = 1.0\n'
                            'arrival_rate = 2\nservice_rate = 4\nfirst_arrival_time = 0\n', encoding='utf-8')
            self.assertEqual(tuple(Scenario.load(path).function_registry.values()), (profiled_function('hello', 2, 4),))

    def test_registry_preserves_configured_function_properties(self):
        examples = Path(__file__).resolve().parents[1] / 'examples/scenarios'
        scenario = Scenario.load(examples / 'startup.toml')
        resize = scenario.function_registry.get('resize')
        self.assertEqual((resize.memory_mb, resize.cpu_demand, resize.max_concurrency), (256, 0.5, 2))
        self.assertEqual(resize.supported_archs, ('amd64',))
        self.assertEqual(resize.profile.service_distribution, 'gamma')
        self.assertIsNone(scenario.function_registry.get('unknown'))
        with self.assertRaises(TypeError):
            scenario.function_registry['other'] = resize

    def test_custom_function_configuration_and_required_resources(self):
        """Il registro conserva immagine e metadati, e richiede risorse esplicite."""
        definition = {
            'name': 'custom-f', 'runtime': 'custom', 'custom_image': 'demo/app:1',
            'memory_mb': 512, 'cpu_demand': 0.75, 'max_concurrency': 2,
            'supported_archs': ['arm64'], 'tag_pattern': 'arm64/demo',
            'tar_function_code': 'app.tar',
            'arrival_rate': 1, 'service_rate': 2, 'first_arrival_time': 0,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scenario.json'
            path.write_text(json.dumps({'functions': [definition]}))
            function = Scenario.load(path).function_registry.get('custom-f')
            self.assertEqual(function.runtime, CustomRuntime('demo/app:1'))
            self.assertEqual(function.max_concurrency, 1)  # Vincolo del runtime custom.
            self.assertEqual(function.supported_archs, ('arm64',))
            self.assertEqual(function.tag_pattern, 'arm64/demo')
            self.assertEqual(function.tar_function_code, 'app.tar')
            missing_arrival = {key: value for key, value in definition.items() if key != 'first_arrival_time'}
            path.write_text(json.dumps({'functions': [missing_arrival]}))
            with self.assertRaisesRegex(ValueError, 'first_arrival_time'):
                Scenario.load(path)
            del definition['memory_mb']
            path.write_text(json.dumps({'functions': [definition]}))
            with self.assertRaisesRegex(ValueError, 'memory_mb'):
                Scenario.load(path)

    def test_configured_distributions_are_used_for_registered_function(self):
        """I campioni deterministici applicano il reciproco dei tassi dichiarati."""
        definition = {
            'name': 'fixed', 'runtime': 'python314', 'memory_mb': 128, 'cpu_demand': 1,
            'arrival_rate': 2, 'service_rate': 4, 'first_arrival_time': 1,
            'arrival_distribution': 'deterministic', 'service_distribution': 'deterministic',
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scenario.json'
            path.write_text(json.dumps({'functions': [definition]}))
            scenario = Scenario.load(path)
            simulation = Simulation(scenario)
            requests = record_submissions(simulation)
            simulation.run(arrivals=3)
            self.assertEqual([request.arrival_time for request in requests], [1, 1.5, 2])
            self.assertEqual([request.service_duration for request in requests], [0.25] * 3)

            definition.update(service_distribution='gamma', service_scv=0.5)
            path.write_text(json.dumps({'functions': [definition]}))
            service = Scenario.load(path).function_registry.get('fixed').sampler.service
            self.assertEqual((service.mean, service.scv), (0.25, 0.5))

    def test_balancer_rejects_at_saturation_even_with_local_queue_enabled(self):
        """I rifiuti del bilanciatore precedono l'ammissione locale."""
        definition = {
            'scheduler': {'queue': {'capacity': 1}},
            'functions': [{
                'name': 'hello', 'runtime': 'python314', 'memory_mb': 128, 'cpu_demand': 1,
                'arrival_rate': 1, 'service_rate': 0.4, 'first_arrival_time': 0,
                'arrival_distribution': 'deterministic', 'service_distribution': 'deterministic',
            }],
            'infrastructure': EDGE,
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'fifo.json'
            path.write_text(json.dumps(definition))
            simulation = Simulation(Scenario.load(path))

        notifications: list[Notification] = []
        self.addCleanup(observer.subscribe(notifications.append))
        simulation.run(arrivals=3)
        engine = simulation._ctx.engine
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual(
            (totals['arrivals'], totals['accepted'], totals['waiting'], totals['queued'], totals['rejected']),
            (3, 1, 0, 0, 2),
        )
        self.assertEqual(totals['pending'], 1)
        self.advance_to(engine, 2.5)
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual((totals['accepted'], totals['completed'], totals['pending']), (1, 1, 0))
        rejections = [event.to_dict()['data'] for event in notifications
                      if event.type is NotificationType.REQUEST_REJECTED]
        self.assertEqual([(event['reason'], event['node_id']) for event in rejections],
                         [('out_of_resources', None)] * 2)

        self.advance_to(engine, 3)
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual((totals['accepted'], totals['warm_starts']), (2, 1))

    def test_local_queue_metrics_follow_queued_and_accepted_notifications(self):
        function = Function('hello', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(
            arrival_rate=1, service_rate=0.4, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))
        simulation = Simulation(Scenario(
            (function,), infrastructure=EDGE, scheduler={'queue': {'capacity': 1}},
            invocation={'entry_node': 'n1'},
        ))
        notifications = []
        self.addCleanup(observer.subscribe(notifications.append))

        totals = simulation.run(arrivals=3).metrics['total']
        self.assertEqual((totals['arrivals'], totals['queued'], totals['waiting'], totals['pending']), (3, 1, 1, 2))
        queued, = [event for event in notifications if event.type is NotificationType.REQUEST_QUEUED]
        self.assertEqual((queued.sim_time_s, queued.region, queued.ring_id, queued.node_id), (1, 'default', 'edge', 'n1'))

        self.advance_to(simulation._ctx.engine, 2.5)
        snapshot = simulation.metrics_snapshot()
        admitted, = [event for event in notifications if event.type is NotificationType.REQUEST_ACCEPTED
                     and event.request_id == queued.request_id]
        dequeued, = [event for event in notifications if event.type is NotificationType.REQUEST_DEQUEUED]
        self.assertEqual(dequeued.request_id, queued.request_id)
        self.assertEqual((dequeued.sim_time_s, dequeued.region, dequeued.ring_id, dequeued.node_id),
                         (2.5, 'default', 'edge', 'n1'))
        self.assertLess(notifications.index(dequeued), notifications.index(admitted))
        self.assertEqual((admitted.queue_wait_s, admitted.warm), (1.5, True))
        totals = snapshot['total']
        self.assertEqual((totals['arrivals'], totals['queued'], totals['waiting'], totals['accepted']), (3, 1, 0, 2))
        self.assertEqual(totals['mean_queue_wait_s'], 1.5)
        self.assertEqual(snapshot['nodes_by_ring']['edge']['n1']['invocations']['mean_queue_wait_s'], 1.5)

    def test_accepted_arrival_completes_and_releases_resources(self):
        """Riuso warm, scarti e prenotazioni confluiscono nelle metriche della replica."""
        function = Function('hello', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(
            arrival_rate=1, service_rate=2 / 3, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))
        scenario = Scenario((function,), infrastructure=EDGE,
                            containers={'idle_timeout': 0.5, 'cleanup_interval': 1})
        simulation = Simulation(scenario)
        requests = record_submissions(simulation)
        simulation.run(arrivals=1)
        engine = simulation._ctx.engine
        first, = requests
        node = simulation._ctx.infrastructure.rings['edge'].nodes[0]
        self.assertEqual(first.function, function)
        self.assertIs(first.node, node)
        self.assertIs(first.status, RequestStatus.ACCEPTED)
        self.assertFalse(first.warm)
        self.assertEqual(node.resources.used_cpus, 1)
        in_flight = simulation.metrics_snapshot()
        self.assertEqual(in_flight['total']['pending'], 1)
        self.assertIsNone(in_flight['total']['mean_response_time_s'])
        self.assertEqual(in_flight['nodes_by_ring']['edge']['n1']['busy_memory_mb'], 128)

        self.assertTrue(engine.step())  # Fine della creazione a t=0.
        self.assertEqual(first.container.state, ContainerState.BUSY)
        self.advance_to(engine, 1)
        rejected = requests[-1]
        self.assertIs(rejected.status, RequestStatus.REJECTED)

        self.advance_to(engine, 1.5)
        self.assertIs(first.status, RequestStatus.COMPLETED)
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(node.resources.used_cpus, 0)
        self.assertEqual(node.resources.warm_pool_used_mem, 128)

        self.advance_to(engine, 2)
        reused = requests[-1]
        self.assertTrue(reused.warm)
        self.assertIs(reused.container, first.container)
        # La prima scadenza è t=2. Al controllo di t=3 il container è
        # di nuovo occupato e rimane disponibile per completare il servizio.
        self.advance_to(engine, 3)
        self.assertTrue(engine.step())  # Pulizia del primo idle a t=3.
        self.assertEqual(engine.now, 3)
        self.assertIs(reused.container.state, ContainerState.BUSY)
        self.assertIsNone(reused.container.expiration_time)
        self.advance_to(engine, 3.5)
        self.assertEqual(reused.container.expiration_time, 4)
        self.assertIs(reused.status, RequestStatus.COMPLETED)
        snapshot = simulation.metrics_snapshot()
        totals = snapshot['total']
        self.assertEqual(snapshot['time_s'], 3.5)
        self.assertEqual(totals, snapshot['by_function']['hello'])
        self.assertEqual((totals['arrivals'], totals['accepted'], totals['completed']), (4, 2, 2))
        self.assertEqual((totals['cold_starts'], totals['warm_starts'], totals['pending']), (1, 1, 0))
        self.assertEqual(totals['rejections_by_reason'], {'out_of_resources': 2})
        self.assertEqual(totals['mean_response_time_s'], 1.5)
        self.assertEqual(totals['mean_service_time_s'], 1.5)
        resources = snapshot['nodes_by_ring']['edge']['n1']
        self.assertEqual((resources['used_cpus'], resources['warm_memory_mb']), (0, 128))
        local = resources['invocations']
        self.assertEqual((local['arrivals'], local['accepted'], local['completed'], local['rejected']), (2, 2, 2, 0))
        self.assertEqual(local['rejections_by_reason'], {})
        self.assertEqual(resources['containers'], totals['containers'])
        self.assertEqual(resources['by_function']['hello'], {**local, 'containers': totals['containers']})
        self.assertEqual(totals['response_time_variance_s2'], 0)
        cpu = resources['time_weighted_resources']['used_cpus']
        self.assertEqual(cpu['observed_time_s'], 3.5)
        self.assertAlmostEqual(cpu['mean'], 6 / 7)
        self.assertAlmostEqual(cpu['variance'], 6 / 49)
        self.assertAlmostEqual(resources['time_weighted_resources']['warm_memory_mb']['mean'], 128 / 7)
        self.assertEqual(in_flight['total']['pending'], 1)

    def test_functions_using_the_same_image_share_startup_time(self):
        # La stessa immagine, selezionata dal runtime integrato o indicata
        # esplicitamente, conserva il proprio costo di avvio. Il costo di
        # caricamento rimane indipendente per ciascuna funzione.
        image = 'grussorusso/serverledge-python314'
        workload = dict(memory_mb=128, cpu_demand=1, arrival_rate=1,
                        service_rate=1, first_arrival_time=0)
        scenario = Scenario([
            dict(workload, name='builtin', runtime='python314', code_load_time=0.125),
            dict(workload, name='second_builtin', runtime='python314', code_load_time=0.5),
            dict(workload, name='custom', runtime='custom', custom_image=image,
                 supported_archs=['amd64'], code_load_time=0.25),
        ], containers={'startup_times': {image: 0.5}})

        builtin, second_builtin, custom = scenario.function_registry.values()
        self.assertEqual(second_builtin.cold_start_time, 1.0)
        self.assertEqual(builtin.runtime.startup_time, 0.5)
        self.assertEqual(custom.runtime.startup_time, 0.5)
        self.assertEqual(builtin.cold_start_time, 0.625)
        self.assertEqual(custom.cold_start_time, 0.75)

    def test_cold_start_then_periodic_cleanup_releases_warm_memory(self):
        """Creazione, servizio e scadenza sono fasi distinte sul calendario."""
        function = Function('hello', 128, 1, BuiltinRuntime('python314', startup_time=0.75), profile=FunctionProfile(
            arrival_rate=0.2, service_rate=2, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
            code_load_time=0.25,
        ))
        scenario = Scenario((function,), infrastructure=EDGE,
                            containers={'idle_timeout': 1.5, 'cleanup_interval': 1})
        simulation = Simulation(scenario)
        requests = record_submissions(simulation)
        notifications: list[Notification] = []
        unsubscribe = observer.subscribe(notifications.append)
        self.addCleanup(unsubscribe)
        simulation.run(arrivals=1)
        engine = simulation._ctx.engine
        first, = requests
        node = first.node
        self.assertIs(first.container.state, ContainerState.CREATING)
        # Una prenotazione non è ancora un'accensione.
        self.assertFalse(any(event.type is NotificationType.CONTAINER_STARTED for event in notifications))
        self.assertEqual(node.resources.used_cpus, 1)
        # Durante il cold start conta una sola istanza, ancora in creazione.
        containers = simulation.metrics_snapshot()['total']['containers']
        self.assertEqual((containers['creating'], containers['creations_started'], containers['created']), (1, 1, 0))
        self.advance_to(engine, 1)
        self.assertIs(first.container.state, ContainerState.BUSY)
        self.assertIs(first.status, RequestStatus.ACCEPTED)
        self.advance_to(engine, 1.5)
        self.assertIs(first.status, RequestStatus.COMPLETED)
        self.assertIs(first.container.state, ContainerState.IDLE)
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual(totals['mean_response_time_s'], 1.5)
        self.assertEqual(totals['mean_service_time_s'], 0.5)
        containers = totals['containers']
        self.assertEqual((containers['creating'], containers['busy'], containers['idle']), (0, 0, 1))
        self.assertEqual(containers['mean_creation_time_s'], 1)
        self.advance_to(engine, 4)
        self.assertIs(first.container.state, ContainerState.DESTROYED)
        self.assertEqual(node.resources.warm_pool_used_mem, 0)
        # Il passaggio in idle non genera uno spegnimento. Il janitor
        # pubblica la rimozione effettiva del container al tempo 4.
        container_events = (NotificationType.CONTAINER_STARTED, NotificationType.CONTAINER_STOPPED)
        started, stopped = [event.to_dict() for event in notifications if event.type in container_events]
        self.assertEqual((started['type'], started['sim_time_s']), (NotificationType.CONTAINER_STARTED, 1))
        self.assertEqual((stopped['type'], stopped['sim_time_s']), (NotificationType.CONTAINER_STOPPED, 4))
        self.assertEqual(started['run_id'], stopped['run_id'])
        for field in ('region', 'ring_id', 'node_id', 'container_id', 'function'):
            self.assertEqual(started['data'][field], stopped['data'][field])
        self.assertEqual(stopped['data']['container_id'], first.container.id)
        self.assertEqual((started['data']['previous'], started['data']['state']), ('creating', 'busy'))
        self.assertEqual((stopped['data']['previous'], stopped['data']['state']), ('idle', 'destroyed'))
        self.assertEqual(started['data']['resources']['busy_memory_mb'], 128)
        self.assertEqual(stopped['data']['resources']['warm_memory_mb'], 0)
        unsubscribe()
        snapshot = simulation.metrics_snapshot()
        node_stats = snapshot['nodes_by_ring']['edge']['n1']
        warm = node_stats['time_weighted_resources']['warm_memory_mb']
        # La rimozione azzera gli idle e conserva il tempo trascorso in quello stato.
        containers = snapshot['total']['containers']
        self.assertEqual((containers['idle'], containers['destroyed']), (0, 1))
        self.assertAlmostEqual(containers['time_weighted_counts']['idle']['mean'], 2.5 / 4)
        self.assertEqual(snapshot['by_function']['hello']['containers'], containers)
        self.assertEqual(node_stats['containers'], containers)
        self.assertEqual(node_stats['by_function']['hello']['containers'], containers)
        self.assertAlmostEqual(warm['mean'], 128 * 2.5 / 4)
        self.advance_to(engine, 5)
        self.assertFalse(requests[-1].warm)
        self.assertIsNot(requests[-1].container, first.container)
        # Il container successivo diventa pronto, ma il subscriber è scollegato.
        self.advance_to(engine, 6)
        self.assertEqual(sum(event.type in container_events for event in notifications), 2)

    def test_scenario_topology_creates_independent_nodes_for_each_replica(self):
        """Le risorse cambiano nella replica avviata, senza modificare la topologia di un'altra."""
        path = Path(__file__).resolve().parents[1] / 'examples/scenarios/startup.toml'
        scenario = Scenario.load(path)
        first, second = Simulation(scenario), Simulation(scenario)
        first.run(arrivals=0)
        second.run(arrivals=0)
        notifications = []
        self.addCleanup(observer.subscribe(notifications.append))
        self.assertTrue(first._ctx.engine.step())
        first_nodes = first.metrics_snapshot()['nodes_by_ring']['edge-x86']
        second_nodes = second.metrics_snapshot()['nodes_by_ring']['edge-x86']
        self.assertEqual(set(first_nodes), {'edge-1', 'edge-2'})
        self.assertEqual(sum(node['used_cpus'] for node in first_nodes.values()), 1)
        self.assertEqual(sum(node['used_cpus'] for node in second_nodes.values()), 0)
        self.assertEqual(first.metrics_snapshot()['total']['accepted'], 1)
        self.assertEqual(second.metrics_snapshot()['total']['arrivals'], 0)
        self.assertTrue(notifications)
        self.assertEqual({event.run_id for event in notifications}, {first.run_id})
        self.assertTrue(second._ctx.engine.step())
        self.assertEqual(second.metrics_snapshot()['total']['accepted'], 1)
        self.assertEqual(first.metrics_snapshot()['total']['arrivals'], 1)


class WelfordTest(unittest.TestCase):
    def test_online_moments_and_timed_snapshot(self):
        moments = Welford()
        for offset in (-2, 0, 2):
            moments.add(10**12 + offset)
        self.assertEqual(moments.mean, 10**12)
        self.assertAlmostEqual(moments.variance, 8 / 3)

        occupancy = TimedWelford(0, 2)
        occupancy.observe(1, 0)
        snapshot = occupancy.snapshot(4)
        self.assertEqual((snapshot.weight, snapshot.mean, snapshot.variance), (4, 0.5, 0.75))
        self.assertEqual(occupancy.snapshot(2).mean, 1)
