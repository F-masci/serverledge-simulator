"""Test end-to-end della simulazione, del ciclo di vita dei container e delle metriche online.

Riferimenti al sistema reale (`serverledge-infrascaler`):
- `internal/lb/lb.go`: pre-ammissione del reverse proxy prima della coda locale del worker.
- `internal/scheduling/policy_default.go` e `internal/scheduling/queue.go`: coda FIFO locale e
  ammissione al completamento.
- `internal/container/runtimes.go` e `internal/container/container.go`: immagini condivise e
  separazione tra startup del container e caricamento del codice.
- `internal/node/pool.go` e `internal/node/janitor_handler.go`: riuso warm e pulizia periodica
  delle istanze idle scadute.
"""

import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario
from simulator.metrics import SimulationMetrics
from simulator.model.execution.container import ContainerState
from simulator.model.execution.function import Function, FunctionProfile
from simulator.model.execution.request import RequestStatus
from simulator.model.execution.runtime import BuiltinRuntime, CustomRuntime
from simulator.observer import Notification, NotificationType, observer
from simulator.utils.welford import TimedWelford, Welford


def profiled_function(name, arrival_rate, service_rate, first_arrival_time=0):
    return Function(
        name, 128, 1.0, BuiltinRuntime('python314'),
        profile=FunctionProfile(arrival_rate, service_rate, first_arrival_time),
    )


EDGE = {'rings': [{
    'id': 'edge', 'arch': 'amd64', 'memory_mb': 128, 'cpus': 1,
    'nodes': [{'name': 'n1'}],
}]}


def record_submissions(simulation):
    requests = []
    simulation.regions['default'].infrastructure.on_request_submitted = (
        lambda infra, request, node: requests.append(request)
    )
    return requests


class SimulationTest(unittest.TestCase):
    def advance_to(self, engine, time):
        """Esegue fino a un istante in cui è previsto almeno un evento."""
        while engine.now < time:
            self.assertTrue(engine.step())
        self.assertEqual(engine.now, time)

    def test_request_notifications_correlate_arrivals_with_outcomes(self):
        """Verifica la correlazione tra arrivi, rifiuti del LB per saturazione e completamenti.

        Riferimento reale: `internal/lb/lb.go` (rifiuto HTTP 429 senza `node_id` a nodo saturo).

        - Con arrivi ogni `1 s` e servizio `2.5 s` su 1 CPU, il primo arrivo completa con
          `response_time_s == 2.5` mentre il secondo viene rifiutato dal LB con `out_of_resources`
          e `node_id=None`, mantenendo la correlazione per `request_id`.
        """
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

    def test_cli_prints_one_structured_report_and_keeps_logs_separate(self):
        """Verifica l'esecuzione end-to-end da CLI (`python -m simulator`) con separazione stdout/stderr.

        - Esegue lo scenario di esempio `examples/scenarios/startup.toml` per 100 arrivi a livelli
          `WARNING` e `INFO`.
        - Verifica che `stdout` contenga esattamente un report strutturato con offloading
          interregionale (`edge → cloud` ed `edge → regional`) e che i log `request.arrived`
          compaiano solo su `stderr` al livello `INFO`.
        """
        root = Path(__file__).resolve().parents[1]
        scenario = 'examples/scenarios/startup.toml'
        command = [sys.executable, '-m', 'simulator', scenario, '--arrivals', '100']
        for level in ('WARNING', 'INFO'):
            with self.subTest(level=level):
                result = subprocess.run(
                    [*command, '--log-level', level], cwd=root,
                    capture_output=True, text=True, check=True, timeout=10,
                )
                self.assertEqual(result.stdout.count('Riepilogo simulazione'), 1)
                self.assertIn(scenario, result.stdout)
                self.assertIn('Invocazioni per funzione', result.stdout)
                self.assertIn('edge (primaria)', result.stdout)
                self.assertIn('Offloading interregionale', result.stdout)
                self.assertRegex(result.stdout, r'(?m)^  edge\s+cloud\s+[1-9]\d*$')
                self.assertRegex(result.stdout, r'(?m)^  edge\s+regional\s+[1-9]\d*$')
                self.assertNotIn('request.arrived', result.stdout)
                self.assertNotIn('simulation.final_statistics', result.stderr)
                self.assertEqual('request.arrived' in result.stderr, level == 'INFO')

    def test_missing_destination_is_a_rejection_without_a_node(self):
        """Verifica che un arrivo in uno scenario privo di nodi venga rifiutato con `no_destination`.

        - Senza infrastruttura configurata, l'arrivo viene notificato e rifiutato immediatamente con
          `reason='no_destination'`, `node_id=None` e `pending=0`.
        """
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
        """Verifica che due repliche con lo stesso `Scenario(seed=42)` producano identici tempi di arrivo e servizio."""
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
        """Verifica che un errore nel collector di metriche interrompa la replica invece di essere ignorato."""
        failure = RuntimeError('metriche non aggiornate')
        scenario = Scenario((profiled_function('hello', 1, 1),))
        with patch.object(SimulationMetrics, 'on_notification', side_effect=failure):
            simulation = Simulation(scenario)
            with self.assertRaises(RuntimeError) as raised:
                simulation.run(arrivals=1)
        self.assertIs(raised.exception, failure)

    def test_example_scenario_formats_define_the_same_workload(self):
        """Verifica che gli scenari di esempio `startup.toml`, `startup.json` e `startup.yaml` siano equivalenti."""
        examples = Path(__file__).resolve().parents[1] / 'examples/scenarios'
        scenario = Scenario.load(examples / 'startup.toml')
        self.assertEqual(scenario.load_balancer, 'Random')
        for suffix in ('json', 'yaml'):
            with self.subTest(format=suffix):
                self.assertEqual(scenario, Scenario.load(examples / f'startup.{suffix}'))

    def test_simulation_seed_controls_ring_selection(self):
        """Verifica che il seed dello scenario controlli in modo riproducibile il routing `Random` tra ring."""
        infrastructure = {'rings': [
            {'id': name, 'arch': 'amd64', 'machine_tag': f'amd64/{name}', 'memory_mb': 512, 'cpus': 2,
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

    def test_arrival_rate_and_service_rate_scaling_and_stream_isolation(self):
        """Verifica il riscalamento proporzionale di `arrival_rate`/`service_rate` e l'isolamento CRN tra funzioni.

        - Raddoppiare `arrival_rate` dimezza gli interarrivi mantenendo `first_arrival_time=1`.
        - Raddoppiare `service_rate` dimezza `service_duration` senza spostare i timestamp di arrivo.
        - Aggiungere una seconda funzione `resize` non perturba la sequenza `(arrival_time, service_duration)`
          della funzione `hello`.
        """
        hello = profiled_function('hello', arrival_rate=2, service_rate=4, first_arrival_time=1)
        faster_arr = replace(hello, profile=replace(hello.profile, arrival_rate=4))
        faster_srv = replace(hello, profile=replace(hello.profile, service_rate=8))

        def collect(functions, arrivals=5):
            sim = Simulation(Scenario(functions, seed=42))
            reqs = record_submissions(sim)
            sim.run(arrivals=arrivals)
            return reqs

        base_reqs = collect((hello,))
        fast_arr_reqs = collect((faster_arr,))
        fast_srv_reqs = collect((faster_srv,))

        self.assertEqual(base_reqs[0].arrival_time, 1)
        self.assertEqual(fast_arr_reqs[0].arrival_time, 1)
        for base, faster in zip(base_reqs[1:], fast_arr_reqs[1:], strict=True):
            self.assertAlmostEqual(base.arrival_time - 1, 2 * (faster.arrival_time - 1))
        for base, faster in zip(base_reqs, fast_srv_reqs, strict=True):
            self.assertEqual(base.arrival_time, faster.arrival_time)
            self.assertAlmostEqual(base.service_duration, 2 * faster.service_duration)

        resize = profiled_function('resize', 3, 8, first_arrival_time=0.5)
        multi_reqs = collect((resize, hello), arrivals=40)
        self.assertEqual(multi_reqs[0].function.name, 'resize')
        hello_multi = [req for req in multi_reqs if req.function.name == 'hello'][:5]
        self.assertEqual(
            [(r.arrival_time, r.service_duration) for r in hello_multi],
            [(r.arrival_time, r.service_duration) for r in base_reqs],
        )

    def test_custom_function_configuration_and_distributions(self):
        """Verifica il caricamento di funzioni `custom` (concorrenza forzata a 1) e distribuzioni configurate.

        Riferimento reale: `internal/api/api.go` (runtime `custom` forza `MaxConcurrency=1` e richiede
        immagine e risorse esplicite).

        - Carica da JSON una funzione `custom` con `max_concurrency=2` e verifica normalizzazione a `1`.
        - Verifica che distribuzioni `deterministic` e `gamma` applichino correttamente i parametri
          dichiarati nello scenario.
        """
        definition = {
            'name': 'custom-f', 'runtime': 'custom', 'custom_image': 'demo/app:1',
            'memory_mb': 512, 'cpu_demand': 0.75, 'max_concurrency': 2,
            'supported_archs': ['arm64'], 'tag_pattern': 'arm64/demo',
            'tar_function_code': 'app.tar',
            'arrival_rate': 2, 'service_rate': 4, 'first_arrival_time': 1,
            'arrival_distribution': 'deterministic', 'service_distribution': 'deterministic',
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'scenario.json'
            path.write_text(json.dumps({'functions': [definition]}), encoding='utf-8')
            scenario = Scenario.load(path)
            fn = scenario.function_registry.get('custom-f')
            self.assertEqual(fn.runtime, CustomRuntime('demo/app:1'))
            self.assertEqual(fn.max_concurrency, 1)
            self.assertEqual(fn.supported_archs, ('arm64',))
            self.assertEqual(fn.tag_pattern, 'arm64/demo')

            sim = Simulation(scenario)
            requests = record_submissions(sim)
            sim.run(arrivals=3)
            self.assertEqual([r.arrival_time for r in requests], [1, 1.5, 2])
            self.assertEqual([r.service_duration for r in requests], [0.25] * 3)

            definition.update(service_distribution='gamma', service_scv=0.5)
            path.write_text(json.dumps({'functions': [definition]}), encoding='utf-8')
            service = Scenario.load(path).function_registry.get('custom-f').sampler.service
            self.assertEqual((service.mean, service.scv), (0.25, 0.5))

    def test_balancer_rejects_at_saturation_even_with_local_queue_enabled(self):
        """Verifica che il LB rifiuti a nodo saturo prima che il worker possa accodare nella FIFO.

        Riferimento reale: `internal/lb/lb.go` (il proxy filtra i target con `NodeMetrics` e
        restituisce HTTP 429 prima di inoltrare la richiesta al worker con coda locale attiva).

        - Nonostante `scheduler.queue.capacity=1`, gli arrivi a `t=1` e `t=2` mentre il nodo è
          saturo (`service=2.5s`) vengono rifiutati dal LB (`node_id=None`, `queued=0`, `rejected=2`).
        - Al termine del servizio (`t=2.5`), il quarto arrivo (`t=3`) viene ammesso come warm start.
        """
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
            path.write_text(json.dumps(definition), encoding='utf-8')
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
        self.advance_to(engine, 2.5)
        rejections = [event.to_dict()['data'] for event in notifications
                      if event.type is NotificationType.REQUEST_REJECTED]
        self.assertEqual([(event['reason'], event['node_id']) for event in rejections],
                         [('out_of_resources', None)] * 2)

        self.advance_to(engine, 3)
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual((totals['accepted'], totals['warm_starts']), (2, 1))

    def test_local_queue_metrics_follow_queued_and_accepted_notifications(self):
        """Verifica che l'invocazione diretta su `entry_node` usi la FIFO locale e tracci correttamente l'attesa in coda.

        Riferimento reale: `internal/scheduling/policy_default.go` e `internal/scheduling/queue.go`.

        - Con `entry_node='n1'` e `queue.capacity=1`, il secondo arrivo (`t=1`) entra in coda
          (`REQUEST_QUEUED`) mentre il terzo (`t=2`) viene rifiutato.
        - A `t=2.5`, al completamento del primo, la richiesta viene estratta (`REQUEST_DEQUEUED`) e
          ammessa warm (`REQUEST_ACCEPTED`) con `queue_wait_s == 1.5` riflesso nelle metriche totali e del nodo.
        """
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
        self.assertEqual((queued.sim_time_s, queued.node.id.key), (1, 'n1'))

        self.advance_to(simulation._ctx.engine, 2.5)
        snapshot = simulation.metrics_snapshot()
        admitted, = [event for event in notifications if event.type is NotificationType.REQUEST_ACCEPTED
                     and event.request is queued.request]
        dequeued, = [event for event in notifications if event.type is NotificationType.REQUEST_DEQUEUED]
        self.assertIs(dequeued.request, queued.request)
        self.assertLess(notifications.index(dequeued), notifications.index(admitted))
        self.assertEqual((admitted.queue_wait_s, admitted.warm), (1.5, True))
        self.assertEqual(snapshot['total']['mean_queue_wait_s'], 1.5)
        self.assertEqual(snapshot['nodes_by_ring']['edge']['n1']['invocations']['mean_queue_wait_s'], 1.5)

    def test_accepted_arrival_completes_and_releases_resources(self):
        """Verifica end-to-end riuso warm, scarti per saturazione e medie temporali pesate (`TimedWelford`).

        Riferimento reale: `internal/node/pool.go` (`AcquireContainer`, `HandleCompletion`) e
        `internal/node/janitor_handler.go` (un container riattivato prima del tick del janitor non
        viene distrutto).

        - Arrivi a `t=0, 1, 2, 3` con `service=1.5s`, `idle_timeout=0.5s`, `cleanup_interval=1s`.
        - `t=0` cold start, `t=1` rifiutato, `t=1.5` completamento e passaggio a idle (scadenza `t=2`),
          `t=2` riattivazione warm che annulla la scadenza al tick del janitor (`t=3`), `t=3.5`
          secondo completamento.
        - Verifica esattezza analitica di medie e varianze temporali di `used_cpus` (`6/7`, `6/49`) e
          `warm_memory_mb` (`128/7`).
        """
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
        node = simulation.regions['default'].infrastructure.resolve_node('n1')
        self.assertFalse(first.warm)
        self.assertEqual(node.resources.used_cpus, 1)

        self.assertTrue(engine.step())  # Fine creazione a t=0.
        self.assertEqual(first.container.state, ContainerState.BUSY)
        self.advance_to(engine, 1)
        self.assertIs(requests[-1].status, RequestStatus.REJECTED)

        self.advance_to(engine, 1.5)
        self.assertIs(first.status, RequestStatus.COMPLETED)
        self.assertEqual((node.resources.used_cpus, node.resources.warm_pool_used_mem), (0, 128))

        self.advance_to(engine, 2)
        reused = requests[-1]
        self.assertTrue(reused.warm)
        self.assertIs(reused.container, first.container)

        self.advance_to(engine, 3)
        self.assertTrue(engine.step())  # Tick janitor a t=3: il container è BUSY e resta vivo.
        self.assertIs(reused.container.state, ContainerState.BUSY)
        self.assertIsNone(reused.container.expiration_time)

        self.advance_to(engine, 3.5)
        self.assertEqual(reused.container.expiration_time, 4)
        self.assertIs(reused.status, RequestStatus.COMPLETED)

        snapshot = simulation.metrics_snapshot()
        totals = snapshot['total']
        self.assertEqual((totals['arrivals'], totals['accepted'], totals['completed']), (4, 2, 2))
        self.assertEqual((totals['cold_starts'], totals['warm_starts'], totals['pending']), (1, 1, 0))
        self.assertEqual(totals['rejections_by_reason'], {'out_of_resources': 2})
        self.assertEqual((totals['mean_response_time_s'], totals['mean_service_time_s']), (1.5, 1.5))

        resources = snapshot['nodes_by_ring']['edge']['n1']
        cpu = resources['time_weighted_resources']['used_cpus']
        self.assertEqual(cpu['observed_time_s'], 3.5)
        self.assertAlmostEqual(cpu['mean'], 6 / 7)
        self.assertAlmostEqual(cpu['variance'], 6 / 49)
        self.assertAlmostEqual(resources['time_weighted_resources']['warm_memory_mb']['mean'], 128 / 7)

    def test_functions_using_the_same_image_share_startup_time(self):
        """Verifica che funzioni con la stessa immagine Docker condividano `startup_time` mantenendo `code_load_time` indipendente.

        Riferimento reale: `internal/container/runtimes.go` e `internal/container/container.go`.

        - Due funzioni `builtin` (`python314`) e una `custom` con la stessa immagine condividono
          `startup_time=0.5s`, sommato al rispettivo `code_load_time` (`0.125`, `0.5`, `0.25`).
        """
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
        self.assertEqual(builtin.runtime.startup_time, 0.5)
        self.assertEqual(custom.runtime.startup_time, 0.5)
        self.assertEqual(builtin.cold_start_time, 0.625)
        self.assertEqual(second_builtin.cold_start_time, 1.0)
        self.assertEqual(custom.cold_start_time, 0.75)

    def test_cold_start_then_periodic_cleanup_releases_warm_memory(self):
        """Verifica la scansione temporale cold start → servizio → idle → pulizia periodica del janitor.

        Riferimento reale: `internal/node/janitor_handler.go` (il janitor gira a intervalli periodici
        `cleanup_interval` e distrugge i container strettamente dopo `expiration_time`).

        - Cold start `1.0s` (`0.75s` startup + `0.25s` code load), servizio `0.5s`, `idle_timeout=1.5s`,
          `cleanup_interval=1s`: `CONTAINER_STARTED` a `t=1`, `COMPLETED`/`IDLE` a `t=1.5` (scadenza `t=3`),
          `CONTAINER_STOPPED` al primo tick strettamente successivo (`t=4`).
        - L'arrivo successivo (`t=5`) richiede un nuovo cold start su un container distinto.
        """
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
        self.assertFalse(any(event.type is NotificationType.CONTAINER_STARTED for event in notifications))

        self.advance_to(engine, 1)
        self.assertIs(first.container.state, ContainerState.BUSY)
        self.advance_to(engine, 1.5)
        self.assertIs(first.status, RequestStatus.COMPLETED)
        self.assertIs(first.container.state, ContainerState.IDLE)
        self.assertEqual(simulation.metrics_snapshot()['total']['mean_response_time_s'], 1.5)
        self.assertEqual(simulation.metrics_snapshot()['total']['containers']['mean_creation_time_s'], 1)

        self.advance_to(engine, 4)
        self.assertIs(first.container.state, ContainerState.DESTROYED)
        self.assertEqual(node.resources.warm_pool_used_mem, 0)
        container_events = (NotificationType.CONTAINER_STARTED, NotificationType.CONTAINER_STOPPED)
        started, stopped = [event.to_dict() for event in notifications if event.type in container_events]
        self.assertEqual((started['sim_time_s'], stopped['sim_time_s']), (1, 4))
        self.assertEqual((started['data']['previous'], started['data']['state']), ('creating', 'busy'))
        self.assertEqual((stopped['data']['previous'], stopped['data']['state']), ('idle', 'destroyed'))

        unsubscribe()
        snapshot = simulation.metrics_snapshot()
        self.assertAlmostEqual(snapshot['total']['containers']['time_weighted_counts']['idle']['mean'], 2.5 / 4)
        self.assertAlmostEqual(
            snapshot['nodes_by_ring']['edge']['n1']['time_weighted_resources']['warm_memory_mb']['mean'],
            128 * 2.5 / 4,
        )

        self.advance_to(engine, 5)
        self.assertFalse(requests[-1].warm)
        self.assertIsNot(requests[-1].container, first.container)

    def test_scenario_topology_creates_independent_nodes_for_each_replica(self):
        """Verifica l'isolamento completo dello stato tra due repliche `Simulation` dello stesso `Scenario`.

        - Avanzare `first` di uno step consuma 1 CPU solo su `first`, lasciando `second` a 0 CPU e
          0 arrivi, con notifiche taggate dal solo `first.run_id`.
        """
        path = Path(__file__).resolve().parents[1] / 'examples/scenarios/startup.toml'
        scenario = Scenario.load(path)
        first, second = Simulation(scenario), Simulation(scenario)
        first.run(arrivals=0)
        second.run(arrivals=0)
        notifications = []
        self.addCleanup(observer.subscribe(notifications.append))

        self.assertTrue(first._ctx.engine.step())
        first_rings = first.metrics_snapshot()['nodes_by_ring']
        second_rings = second.metrics_snapshot()['nodes_by_ring']
        first_nodes = {**first_rings['edge-x86'], **first_rings['edge-x86-standard']}
        second_nodes = {**second_rings['edge-x86'], **second_rings['edge-x86-standard']}
        self.assertEqual(sum(node['used_cpus'] for node in first_nodes.values()), 1)
        self.assertEqual(sum(node['used_cpus'] for node in second_nodes.values()), 0)
        self.assertEqual({event.run_id for event in notifications}, {first.run_id})

        self.assertTrue(second._ctx.engine.step())
        self.assertEqual(second.metrics_snapshot()['total']['accepted'], 1)
        self.assertEqual(first.metrics_snapshot()['total']['arrivals'], 1)

    def test_scale_in_during_cold_start_drains_cleanly_and_updates_topology(self):
        """Verifica che uno scale-in durante un cold start dreni la richiesta in volo senza errori.

        - Rimuove il nodo `n1` mentre il container è ancora in `CREATING` (`current_active_nodes=0`,
          `scale_in_events=1`).
        - Avanzando il calendario fino a `t=1.5`, il cold start e l'esecuzione completano regolarmente
          (`accepted=1`, `completed=1`, `pending=0`).
        """
        function = Function('hello', 128, 1, BuiltinRuntime('python314', startup_time=1.0), profile=FunctionProfile(
            arrival_rate=0.1, service_rate=2, first_arrival_time=0,
            arrival_distribution='deterministic', service_distribution='deterministic',
        ))
        simulation = Simulation(Scenario((function,), infrastructure=EDGE))
        simulation.run(arrivals=1)
        engine = simulation._ctx.engine
        infra = simulation.regions['default'].infrastructure

        infra.remove_node('edge', 'n1')
        topology = simulation.metrics_snapshot()['topology']
        self.assertEqual((topology['current_active_nodes'], topology['scale_in_events']), (0, 1))

        self.advance_to(engine, 1.5)
        totals = simulation.metrics_snapshot()['total']
        self.assertEqual((totals['accepted'], totals['completed'], totals['pending']), (1, 1, 0))


class WelfordTest(unittest.TestCase):
    def test_online_moments_and_timed_snapshot(self):
        """Verifica stabilità numerica di `Welford` (inclusi pesi 0, positivi e rifiuto pesi negativi) e `TimedWelford`."""
        moments = Welford()
        for offset in (-2, 0, 2):
            moments.add(10**12 + offset)
        self.assertEqual(moments.mean, 10**12)
        self.assertAlmostEqual(moments.variance, 8 / 3)
        self.assertEqual(moments.weight, 3.0)

        weighted = Welford()
        weighted.add(10.0, weight=0.0)
        self.assertIsNone(weighted.mean)
        weighted.add(10.0, weight=2.0)
        weighted.add(20.0, weight=2.0)
        self.assertEqual(weighted.weight, 4.0)
        self.assertAlmostEqual(weighted.mean, 15.0)
        with self.assertRaises(ValueError):
            weighted.add(1.0, weight=-1.0)

        occupancy = TimedWelford(0, 2)
        occupancy.observe(1, 0)
        snapshot = occupancy.snapshot(4)
        self.assertEqual((snapshot.weight, snapshot.mean, snapshot.variance), (4, 0.5, 0.75))
        self.assertEqual(occupancy.snapshot(2).mean, 1)

    def test_cli_path_validator_rejects_traversal_outside_cwd(self):
        """Verifica che la validazione canonica del percorso CLI blocchi path traversal fuori dalla directory corrente."""
        from simulator.__main__ import _resolve_scenario_path

        with self.assertRaisesRegex(ValueError, 'fuori dalla directory'):
            _resolve_scenario_path(Path('/etc/passwd'))


if __name__ == '__main__':
    unittest.main()
