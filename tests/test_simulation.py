"""Verifiche degli arrivi e del ciclo di vita dei container.

Riferimenti Infrascaler:
- internal/scheduling/policy_default.go: FIFO opzionale e ripartenza al completamento.
- internal/scheduling/queue.go: ordine di accodamento e capacità.
- internal/container/runtimes.go: immagini Docker dei runtime integrati.
- internal/container/container.go: scelta dell'immagine e copia del codice.
- internal/node/pool.go: riuso del container e scadenza delle istanze idle.
- internal/node/janitor_handler.go: pulizia periodica dei container scaduti.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sim.core import EventStatus
from sim.io.scenario import Scenario
from sim.model.function import BuiltinRuntime, CustomRuntime, Function, FunctionProfile
from sim.core.simulation import Simulation
from sim.model.events import ArrivalEvent
from sim.model.container import ContainerState
from sim.model.infrastructure import Infrastructure
from sim.model.request import RequestStatus
from sim.observer import Notification, NotificationType, SimulationError, observer
from sim.observer.logger import log_notification
from sim.utils.welford import TimedWelford, Welford


def profiled_function(name, arrival_rate, service_rate, first_arrival_time=0):
    return Function(name, 128, 1.0, BuiltinRuntime('python314'),
                    profile=FunctionProfile(arrival_rate, service_rate, first_arrival_time))


class RecordingInfrastructure(Infrastructure):
    def __init__(self):
        super().__init__()
        self.requests = []

    def submit(self, request):
        self.requests.append(request)
        return super().submit(request)


class SimulationTest(unittest.TestCase):
    def test_request_notifications_correlate_arrivals_with_outcomes(self):
        """Il primo arrivo completa e il secondo viene rifiutato dal nodo saturo."""
        function = Function('hello', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(
            arrival_rate=1, service_rate=0.4, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))
        scenario = Scenario((function,), infrastructure={'rings': [{
            'id': 'edge', 'arch': 'amd64', 'memory_mb': 128, 'cpus': 1,
            'nodes': [{'name': 'n1', 'area': 'edge'}],
        }]})
        notifications: list[Notification] = []
        self.addCleanup(observer.subscribe(notifications.append))

        Simulation(scenario).run(arrivals=4)

        payloads = [json.loads(json.dumps(event.to_dict())) for event in notifications
                    if event.type.value.startswith('request.')]
        first, second, rejected = [payload['data'] for payload in payloads[:3]]
        self.assertEqual([payload['type'] for payload in payloads[:3]],
                         ['request.arrived', 'request.arrived', 'request.rejected'])
        self.assertEqual(rejected['request_id'], second['request_id'])
        self.assertEqual(rejected['reason'], 'out_of_resources')
        completed, = [payload['data'] for payload in payloads if payload['type'] == 'request.completed']
        self.assertEqual(completed['request_id'], first['request_id'])
        self.assertEqual(completed['response_time_s'], 2.5)

    def test_final_statistics_can_be_emitted_after_run(self):
        """Il chiamante pubblica lo snapshot serializzabile dopo il ritorno di run."""
        function = profiled_function('hello', 1, 1, first_arrival_time=2)
        simulation = Simulation(Scenario((function,)))
        notifications: list[Notification] = []
        self.addCleanup(observer.subscribe(notifications.append))

        self.assertIs(simulation.run(arrivals=1), simulation.metrics)
        self.assertFalse(any(event.type is NotificationType.FINAL_STATISTICS for event in notifications))
        observer.notify(simulation.final_statistics())

        event, = [event for event in notifications if event.type is NotificationType.FINAL_STATISTICS]
        payload = json.loads(json.dumps(event.to_dict()))
        self.assertEqual(payload['type'], NotificationType.FINAL_STATISTICS)
        self.assertEqual(payload['data']['outcome'], 'completed')
        self.assertEqual(payload['data']['metrics'], simulation.metrics_snapshot())
        self.assertEqual(payload['sim_time_s'], 2)

    def test_failed_calendar_event_is_emitted_and_logged_as_error(self):
        """L'errore raggiunge observer e logger, poi torna al chiamante."""
        failure = ValueError('inoltro fallito')

        class FailingInfrastructure(Infrastructure):
            def submit(self, request):
                raise failure

        function = profiled_function('hello', 1, 1, first_arrival_time=2)
        simulation = Simulation(Scenario((function,)), FailingInfrastructure())
        occurrence, = simulation.start()
        notifications: list[Notification] = []
        self.addCleanup(observer.subscribe(notifications.append))
        self.addCleanup(observer.subscribe(log_notification))

        with self.assertLogs('sim.notifications', level='ERROR') as logs:
            with self.assertRaises(ValueError) as raised:
                simulation.engine.step()
        self.assertIs(raised.exception, failure)
        self.assertIs(occurrence.status, EventStatus.FAILED)
        event, = [event for event in notifications if event.type is NotificationType.SIMULATION_ERROR]
        self.assertIsInstance(event, SimulationError)
        self.assertIs(event.exception, failure)
        payload = json.loads(json.dumps(event.to_dict()))
        self.assertEqual(payload['type'], 'simulation.error')
        self.assertEqual(payload['sim_time_s'], 2)
        self.assertEqual(payload['data'], {
            'failed_event': 'ArrivalEvent',
            'error_type': 'ValueError',
            'message': 'inoltro fallito',
        })
        self.assertIn('ValueError: inoltro fallito', logs.output[0])

    def test_each_arrival_leaves_one_successor_in_the_calendar(self):
        """L'avvio lascia l'orologio a zero, poi ogni step pianifica un arrivo futuro."""
        function = profiled_function('hello', arrival_rate=2.0, service_rate=4.0, first_arrival_time=2.0)
        simulation = Simulation(Scenario(functions=(function,)))
        initial, = simulation.start()
        engine = simulation.engine
        self.assertIsInstance(initial.event, ArrivalEvent)
        self.assertEqual(initial.event.function, function)
        self.assertIs(initial.status, EventStatus.PENDING)
        self.assertEqual((engine.now, simulation.arrivals), (0, 0))
        self.assertEqual(engine.next_event_time, 2.0)

        for count in range(1, 11):
            scheduled_time = engine.next_event_time
            self.assertTrue(engine.step())
            self.assertEqual(engine.now, scheduled_time)
            self.assertEqual(simulation.arrivals, count)
            self.assertEqual(engine.pending_count, 1)
            self.assertGreater(engine.next_event_time, engine.now)
        self.assertIs(initial.status, EventStatus.EXECUTED)

    def test_same_seed_reproduces_arrival_times(self):
        """Due repliche avviate con lo stesso scenario avanzano indipendentemente."""
        function = profiled_function('hello', arrival_rate=3, service_rate=4)
        scenario = Scenario(functions=(function,), seed=42)
        first, second = Simulation(scenario), Simulation(scenario)
        first.start()
        second.start()
        for _ in range(5):
            first.engine.step()
            second.engine.step()
            self.assertEqual(first.engine.next_event_time, second.engine.next_event_time)

    def test_scenario_profile_controls_the_arrival_process(self):
        """Il profilo letto da file arriva all'evento e il tasso scala l'interarrivo."""
        examples = Path(__file__).resolve().parents[1] / 'examples/scenarios'
        scenario = Scenario.load(examples / 'startup.toml')
        self.assertEqual(scenario, Scenario.load(examples / 'startup.json'))
        function = scenario.get_function('hello')
        profile = function.profile
        baseline = Simulation(Scenario((function,), seed=scenario.seed))
        twice_as_fast = Simulation(Scenario(
            (profiled_function(function.name, 2 * profile.arrival_rate, profile.service_rate, profile.first_arrival_time),),
            seed=scenario.seed,
        ))
        for simulation in (baseline, twice_as_fast):
            first, = simulation.start()
            self.assertEqual(first.event.function.profile.service_rate, 4.0)
            simulation.engine.step()
        self.assertAlmostEqual(
            baseline.engine.next_event_time - baseline.engine.now,
            2 * (twice_as_fast.engine.next_event_time - twice_as_fast.engine.now),
        )

    def test_function_profile_rejects_invalid_rates(self):
        for field in ('arrival_rate', 'service_rate'):
            for value in (0, -1, float('inf'), True):
                with self.subTest(field=field, value=value):
                    rates = {'arrival_rate': 2, 'service_rate': 4, 'first_arrival_time': 0, field: value}
                    with self.assertRaises(ValueError):
                        FunctionProfile(**rates)

    def test_service_rate_scales_service_duration_without_shifting_arrivals(self):
        """A parità di seed, raddoppiare il tasso dimezza i campioni di servizio."""
        baseline_target, faster_target = RecordingInfrastructure(), RecordingInfrastructure()
        baseline = Simulation(Scenario((profiled_function('hello', 2, 4),)), baseline_target)
        faster = Simulation(Scenario((profiled_function('hello', 2, 8),)), faster_target)
        for simulation in (baseline, faster):
            simulation.start()

        for _ in range(5):
            baseline.engine.step()
            faster.engine.step()
            self.assertEqual(baseline.engine.next_event_time, faster.engine.next_event_time)
            baseline_duration = baseline_target.requests[-1].service_duration
            self.assertGreater(baseline_duration, 0)
            self.assertAlmostEqual(baseline_duration, 2 * faster_target.requests[-1].service_duration)

    def test_functions_keep_independent_arrival_and_service_streams(self):
        """Intercalare gli arrivi di un'altra funzione lascia invariata la sequenza di hello."""
        hello = profiled_function('hello', 2, 4, first_arrival_time=1)
        resize = profiled_function('resize', 3, 8, first_arrival_time=0.5)
        single_target, multiple_target = RecordingInfrastructure(), RecordingInfrastructure()
        single = Simulation(Scenario((hello,)), single_target)
        multiple = Simulation(Scenario((resize, hello)), multiple_target)
        single.start()
        initial = multiple.start()
        self.assertEqual([event.event.function for event in initial], [resize, hello])
        self.assertEqual([event.time for event in initial], [0.5, 1])
        self.assertEqual(multiple.engine.pending_count, 2)

        for _ in range(5):
            single.engine.step()
            for _ in range(100):
                self.assertTrue(multiple.engine.step())
                self.assertEqual(multiple.engine.pending_count, 2)
                if multiple_target.requests[-1].function.name == hello.name:
                    break
            self.assertEqual(multiple_target.requests[-1].function, hello)
            self.assertEqual(multiple.engine.now, single.engine.now)
            self.assertEqual(multiple_target.requests[-1].service_duration,
                             single_target.requests[-1].service_duration)
        self.assertTrue(any(request.function.name == resize.name for request in multiple_target.requests))

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
            self.assertEqual(Scenario.load(path).functions, (profiled_function('hello', 2, 4),))

    def test_registry_contains_configured_functions_used_by_arrival_events(self):
        """L'evento riceve una copia per replica della funzione registrata con le risorse da file."""
        examples = Path(__file__).resolve().parents[1] / 'examples/scenarios'
        scenario = Scenario.load(examples / 'startup.toml')
        resize = scenario.get_function('resize')
        self.assertEqual((resize.memory_mb, resize.cpu_demand, resize.max_concurrency), (256, 0.5, 2))
        self.assertEqual(resize.supported_archs, ('amd64',))
        self.assertEqual(resize.profile.service_distribution, 'gamma')
        self.assertIsNone(scenario.get_function('unknown'))
        with self.assertRaises(TypeError):
            scenario.function_registry['other'] = resize
        _, second = Simulation(scenario).start()
        self.assertEqual(second.event.function, resize)
        self.assertIsNot(second.event.function.sampler, resize.sampler)

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
            function = Scenario.load(path).get_function('custom-f')
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
            target = RecordingInfrastructure()
            simulation = Simulation(scenario, target)
            simulation.start()
            for expected_time in (1, 1.5, 2):
                self.assertTrue(simulation.engine.step())
                self.assertEqual(simulation.engine.now, expected_time)
                self.assertEqual(target.requests[-1].service_duration, 0.25)

            definition.update(service_distribution='gamma', service_scv=0.5)
            path.write_text(json.dumps({'functions': [definition]}))
            service = Scenario.load(path).get_function('fixed').sampler.service
            self.assertEqual((service.mean, service.scv), (0.25, 0.5))

    def test_configured_fifo_counts_waiting_and_retries_after_completion(self):
        """La FIFO configurata attende un completamento e scarta quando è piena."""
        definition = {
            'scheduler': {'queue': {'capacity': 1}},
            'functions': [{
                'name': 'hello', 'runtime': 'python314', 'memory_mb': 128, 'cpu_demand': 1,
                'arrival_rate': 1, 'service_rate': 0.4, 'first_arrival_time': 0,
                'arrival_distribution': 'deterministic', 'service_distribution': 'deterministic',
            }],
            'infrastructure': {'rings': [{
                'id': 'edge', 'arch': 'amd64', 'memory_mb': 128, 'cpus': 1,
                'nodes': [{'name': 'n1', 'area': 'edge'}],
            }]},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'fifo.json'
            path.write_text(json.dumps(definition))
            simulation = Simulation(Scenario.load(path))

        simulation.start()
        simulation.engine.run(until=2)
        totals = simulation.metrics_snapshot()['total']
        # A t=2 la prima richiesta è attiva, la seconda attende e la terza è scartata.
        self.assertEqual((totals['arrivals'], totals['accepted'], totals['waiting'], totals['rejected']),
                         (3, 1, 1, 1))
        self.assertEqual(totals['pending'], 2)
        simulation.engine.run(until=2.5)
        totals = simulation.metrics_snapshot()['total']
        # Il primo completamento libera il container e avvia la testa con warm start.
        self.assertEqual((totals['arrivals'], totals['accepted'], totals['waiting'], totals['warm_starts']),
                         (3, 2, 0, 1))
        self.assertEqual(totals['mean_queue_wait_s'], 1.5)
        simulation.engine.run(until=5)
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual(totals['completed'], 2)
        self.assertEqual(totals['mean_response_time_s'], 3.25)

    def test_accepted_arrival_completes_and_releases_resources(self):
        """Uno scarto non schedula completamenti e il riuso invalida la vecchia pulizia."""
        function = Function('hello', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(
            arrival_rate=1, service_rate=2 / 3, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))
        target = RecordingInfrastructure()
        target.create_ring('edge', arch='amd64', memory=128, cpus=1)
        node = target.create_node('edge', 'n1', 'edge', idle_timeout=0.5)
        scenario = Scenario((function,), containers={'cleanup_interval': 1})
        simulation = Simulation(scenario, target)
        simulation.start()
        engine = simulation.engine
        engine.step()
        first = target.requests[-1]
        self.assertEqual(first.function, function)
        self.assertIs(first.node, node)
        self.assertIs(first.status, RequestStatus.ACCEPTED)
        self.assertFalse(first.warm)
        self.assertEqual(node.resources.used_cpus, 1)
        self.assertEqual(engine.pending_count, 2)
        in_flight = simulation.metrics_snapshot()
        self.assertEqual(in_flight['total']['pending'], 1)
        self.assertIsNone(in_flight['total']['mean_response_time_s'])
        self.assertEqual(in_flight['nodes_by_ring']['edge']['n1']['busy_memory_mb'], 128)

        engine.step()  # Fine della creazione a t=0.
        self.assertEqual(first.container.state, ContainerState.BUSY)
        engine.step()
        self.assertEqual(engine.now, 1)
        rejected = target.requests[-1]
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(engine.pending_count, 2)
        self.assertEqual(engine.next_event_time, 1.5)

        engine.step()
        self.assertEqual(engine.now, 1.5)
        self.assertIs(first.status, RequestStatus.COMPLETED)
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(node.resources.used_cpus, 0)
        self.assertEqual(node.resources.warm_pool_used_mem, 128)
        self.assertEqual(engine.pending_count, 2)

        engine.step()
        reused = target.requests[-1]
        self.assertTrue(reused.warm)
        self.assertIs(reused.container, first.container)
        # La prima scadenza è t=2. Al controllo di t=3 il container è
        # di nuovo occupato e rimane disponibile per completare il servizio.
        engine.run(until=3)
        self.assertIs(reused.container.state, ContainerState.BUSY)
        self.assertIsNone(reused.container.expiration_time)
        engine.run(until=3.5)
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
        self.assertEqual(resources['invocations'], {key: value for key, value in totals.items() if key != 'containers'})
        self.assertEqual(resources['containers'], totals['containers'])
        self.assertEqual(resources['by_function']['hello'], totals)
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
            dict(workload, name='custom', runtime='custom', custom_image=image,
                 supported_archs=['amd64'], code_load_time=0.25),
        ], containers={'startup_times': {image: 0.5}})

        builtin, custom = scenario.functions
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
        target = RecordingInfrastructure()
        target.create_ring('edge', arch='amd64', memory=128, cpus=1)
        node = target.create_node('edge', 'n1', 'edge', idle_timeout=1.5)
        scenario = Scenario((function,), containers={'cleanup_interval': 1})
        simulation = Simulation(scenario, target)
        notifications: list[Notification] = []
        unsubscribe = observer.subscribe(notifications.append)
        self.addCleanup(unsubscribe)
        simulation.start()
        engine = simulation.engine
        engine.run(until=0.5)
        first = target.requests[0]
        self.assertIs(first.container.state, ContainerState.CREATING)
        # Una prenotazione non è ancora un'accensione.
        self.assertFalse(any(event.type is NotificationType.CONTAINER_STARTED for event in notifications))
        self.assertEqual(node.resources.used_cpus, 1)
        # Durante il cold start conta una sola istanza, ancora in creazione.
        containers = simulation.metrics_snapshot()['total']['containers']
        self.assertEqual((containers['creating'], containers['creations_started'], containers['created']), (1, 1, 0))
        engine.run(until=1)
        self.assertIs(first.container.state, ContainerState.BUSY)
        self.assertIs(first.status, RequestStatus.ACCEPTED)
        engine.run(until=1.5)
        self.assertIs(first.status, RequestStatus.COMPLETED)
        self.assertIs(first.container.state, ContainerState.IDLE)
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual(totals['mean_response_time_s'], 1.5)
        self.assertEqual(totals['mean_service_time_s'], 0.5)
        containers = totals['containers']
        self.assertEqual((containers['creating'], containers['busy'], containers['idle']), (0, 0, 1))
        self.assertEqual(containers['mean_creation_time_s'], 1)
        engine.run(until=3)
        self.assertIs(first.container.state, ContainerState.IDLE)
        engine.run(until=4)
        self.assertIs(first.container.state, ContainerState.DESTROYED)
        self.assertEqual(node.resources.warm_pool_used_mem, 0)
        engine.run(until=4.5)
        # Il passaggio in idle non genera uno spegnimento. Il janitor
        # pubblica la rimozione effettiva del container al tempo 4.
        container_events = (NotificationType.CONTAINER_STARTED, NotificationType.CONTAINER_STOPPED)
        started, stopped = [event.to_dict() for event in notifications if event.type in container_events]
        self.assertEqual((started['type'], started['sim_time_s']), (NotificationType.CONTAINER_STARTED, 1))
        self.assertEqual((stopped['type'], stopped['sim_time_s']), (NotificationType.CONTAINER_STOPPED, 4))
        self.assertEqual(started['run_id'], stopped['run_id'])
        self.assertEqual(started['data'], stopped['data'])
        self.assertEqual(stopped['data']['container_id'], first.container.id)
        unsubscribe()
        snapshot = simulation.metrics_snapshot()
        node_stats = snapshot['nodes_by_ring']['edge']['n1']
        warm = node_stats['time_weighted_resources']['warm_memory_mb']
        # La rimozione azzera gli idle e conserva il tempo trascorso in quello stato.
        containers = snapshot['total']['containers']
        self.assertEqual((containers['idle'], containers['destroyed']), (0, 1))
        self.assertAlmostEqual(containers['time_weighted_counts']['idle']['mean'], 2.5 / 4.5)
        self.assertEqual(snapshot['by_function']['hello']['containers'], containers)
        self.assertEqual(node_stats['containers'], containers)
        self.assertEqual(node_stats['by_function']['hello']['containers'], containers)
        self.assertAlmostEqual(warm['mean'], 128 * 2.5 / 4.5)
        engine.run(until=5)
        self.assertFalse(target.requests[-1].warm)
        self.assertIsNot(target.requests[-1].container, first.container)
        # Il container successivo diventa pronto, ma il subscriber è scollegato.
        engine.run(until=6)
        self.assertEqual(sum(event.type in container_events for event in notifications), 2)

    def test_scenario_topology_creates_independent_nodes_for_each_replica(self):
        """Le risorse cambiano nella replica avviata, senza modificare la topologia di un'altra."""
        path = Path(__file__).resolve().parents[1] / 'examples/scenarios/startup.toml'
        scenario = Scenario.load(path)
        first, second = Simulation(scenario), Simulation(scenario)
        first.start()
        second.start()
        first.engine.step()
        first_nodes = first.metrics_snapshot()['nodes_by_ring']['edge-x86']
        second_nodes = second.metrics_snapshot()['nodes_by_ring']['edge-x86']
        self.assertEqual(set(first_nodes), {'edge-1', 'edge-2'})
        self.assertEqual(sum(node['used_cpus'] for node in first_nodes.values()), 1)
        self.assertEqual(sum(node['used_cpus'] for node in second_nodes.values()), 0)
        self.assertEqual(first.metrics_snapshot()['total']['accepted'], 1)


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
