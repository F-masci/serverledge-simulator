"""Verifiche della pianificazione degli arrivi a orizzonte infinito."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sim.core import EventStatus
from sim.io.scenario import Scenario
from sim.model.function import BuiltinRuntime, CustomRuntime, Function, FunctionProfile
from sim.core.simulation import Simulation
from sim.model.events import ArrivalEvent
from sim.model.infrastructure import Infrastructure
from sim.model.request import RequestStatus


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

    def test_accepted_arrival_completes_and_releases_resources(self):
        """Uno scarto lascia in calendario solo il completamento già previsto e il prossimo arrivo."""
        function = Function('hello', 128, 1, BuiltinRuntime('python314'), profile=FunctionProfile(
            arrival_rate=1, service_rate=2 / 3, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))
        target = RecordingInfrastructure()
        target.create_ring('edge', arch='amd64', memory=128, cpus=1)
        node = target.create_node('edge', 'n1', 'edge')
        simulation = Simulation(Scenario((function,)), target)
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
        self.assertEqual(engine.pending_count, 1)

        engine.step()
        reused = target.requests[-1]
        self.assertTrue(reused.warm)
        self.assertIs(reused.container, first.container)
        engine.run(until=3.5)
        self.assertIs(reused.status, RequestStatus.COMPLETED)
