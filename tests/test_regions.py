"""Ingressi regionali e trasferimenti tra infrastrutture indipendenti."""

from pathlib import Path
import unittest

from sim.core.simulation import Simulation
from sim.io.scenario import Scenario
from sim.model.request import Request, RequestStatus
from sim.observer import NotificationType, observer


FUNCTION = {
    'name': 'f', 'runtime': 'python314', 'memory_mb': 128, 'cpu_demand': 1,
    'arrival_rate': 0.1, 'service_rate': 0.5, 'first_arrival_time': 0,
    'arrival_distribution': 'deterministic', 'service_distribution': 'deterministic',
}


def simulated_region(*, latency=0, **node_settings):
    return {
        'offload_latency_s': latency,
        'infrastructure': {'rings': [{
            'id': 'amd64', 'arch': 'amd64', 'memory_mb': 128, 'cpus': 1,
            'nodes': [{'name': 'worker', 'area': 'local', **node_settings}],
        }]},
    }


class RegionsTest(unittest.TestCase):
    def setUp(self):
        self.notifications = []
        self.addCleanup(observer.subscribe(self.notifications.append))

    def advance_to(self, simulation, time):
        engine = simulation._ctx.engine
        while engine.now < time:
            self.assertTrue(engine.step())
        self.assertEqual(engine.now, time)

    def test_offload_to_always_successful_region_counts_one_arrival_and_round_trip(self):
        path = Path(__file__).resolve().parents[1] / 'examples/scenarios/regional-offloading.toml'
        simulation = Simulation(Scenario.load(path))

        initial = simulation.run(arrivals=1).metrics
        self.assertEqual(initial['primary_region'], 'edge')
        self.assertEqual(initial['total']['arrivals'], 1)
        self.assertEqual(initial['total']['in_transit'], 1)
        self.assertEqual(initial['regions']['edge']['external_arrivals'], 1)
        self.assertEqual(initial['regional_offloads'], [{'source': 'edge', 'destination': 'cloud', 'count': 1}])
        self.assertIsNone(simulation.regions['cloud'].infrastructure)

        self.advance_to(simulation, 0.125)
        accepted = simulation.metrics_snapshot()['total']
        self.assertEqual((accepted['arrivals'], accepted['accepted'], accepted['in_transit']), (1, 1, 0))
        self.assertEqual(accepted['accepted_without_container'], 1)

        self.advance_to(simulation, 0.625)
        snapshot = simulation.metrics_snapshot()
        self.assertEqual(snapshot['total']['completed'], 1)
        self.assertEqual(snapshot['total']['mean_response_time_s'], 0.625)
        self.assertEqual(snapshot['total']['mean_service_time_s'], 0.375)
        self.assertEqual(snapshot['total']['warm_starts'] + snapshot['total']['cold_starts'], 0)
        self.assertEqual(snapshot['regions']['cloud']['nodes_by_ring'], {})
        self.assertEqual(snapshot['regions']['cloud']['invocations']['completed'], 1)

        arrived, admitted, completed = self.notifications
        self.assertIs(arrived.type, NotificationType.REQUEST_ARRIVED)
        self.assertIs(admitted.type, NotificationType.REQUEST_ACCEPTED)
        self.assertIs(completed.type, NotificationType.REQUEST_COMPLETED)
        self.assertEqual({event.request_id for event in self.notifications}, {arrived.request_id})
        self.assertEqual((arrived.region, completed.region), ('edge', 'cloud'))
        self.assertIsNone(completed.node_id)
        self.assertIsNone(completed.container_id)

    def test_offload_to_simulated_region_releases_remote_resources_before_return(self):
        scenario = Scenario([FUNCTION], primary_region='edge', invocation={'can_offload': True}, regions={
            'edge': simulated_region(policy='cloudonly', remote_region='cloud'),
            'cloud': simulated_region(latency=4),
        })
        simulation = Simulation(scenario)
        simulation.run(arrivals=1)
        self.advance_to(simulation, 2)
        snapshot = simulation.metrics_snapshot()
        edge_node = snapshot['nodes_by_ring']['amd64']['worker']
        cloud_node = snapshot['regions']['cloud']['nodes_by_ring']['amd64']['worker']
        self.assertEqual(edge_node['used_cpus'], 0)
        self.assertEqual(cloud_node['used_cpus'], 1)

        self.advance_to(simulation, 4)
        snapshot = simulation.metrics_snapshot()
        self.assertEqual(snapshot['regions']['cloud']['nodes_by_ring']['amd64']['worker']['used_cpus'], 0)
        self.assertEqual((snapshot['total']['completed'], snapshot['total']['pending']), (0, 1))

        self.advance_to(simulation, 6)
        final = simulation.metrics_snapshot()['total']
        self.assertEqual((final['arrivals'], final['completed'], final['pending']), (1, 1, 0))
        self.assertEqual(final['mean_response_time_s'], 6)
        self.assertEqual(final['cold_starts'], 1)

    def test_offload_uses_source_specific_round_trip_or_destination_default(self):
        sources = ('north', 'local', 'south')
        regions = {source: simulated_region(policy='cloudonly', remote_region='cloud') for source in sources}
        regions['cloud'] = {
            'mode': 'always_succeeds',
            'service_time_s': 0.5,
            'offload_latency_s': 4,
            'offload_latency_by_source_s': {'north': 2, 'local': 0},
        }
        scenario = Scenario([dict(FUNCTION, first_arrival_time=10)], regions=regions)
        simulation = Simulation(scenario)
        simulation.run(arrivals=0)

        requests = {}
        for source in sources:
            request = Request(scenario.function_registry['f'], can_offload=True)
            requests[request.id] = source
            simulation.submit(request, region=source)
        self.advance_to(simulation, 4.5)

        admissions = {requests[event.request_id]: event.sim_time_s for event in self.notifications
                      if event.type is NotificationType.REQUEST_ACCEPTED}
        completions = {requests[event.request_id]: event.response_time_s for event in self.notifications
                       if event.type is NotificationType.REQUEST_COMPLETED}
        self.assertEqual(admissions, {'north': 1, 'local': 0, 'south': 2})
        self.assertEqual(completions, {'north': 2.5, 'local': 0.5, 'south': 4.5})

    def test_cannot_offload_again_from_a_simulated_destination(self):
        scenario = Scenario([FUNCTION], invocation={'can_offload': True}, regions={
            'edge': simulated_region(policy='cloudonly', remote_region='relay'),
            'relay': simulated_region(policy='cloudonly', remote_region='cloud'),
            'cloud': {'mode': 'always_succeeds'},
        })
        simulation = Simulation(scenario)
        requests = []
        simulation.regions['relay'].infrastructure.on_request_submitted = (
            lambda _, request, node: requests.append(request)
        )
        stats = simulation.run(arrivals=1).metrics
        request, = requests
        self.assertIs(request.status, RequestStatus.REJECTED)
        self.assertFalse(request.can_offload)
        self.assertEqual((request.entry_region, request.region), ('edge', 'relay'))
        self.assertEqual(stats['total']['rejected'], 1)
        self.assertEqual(stats['regional_offloads'], [{'source': 'edge', 'destination': 'relay', 'count': 1}])

    def test_entry_region_can_override_primary_and_complete_immediately(self):
        scenario = Scenario([FUNCTION], primary_region='edge', regions={
            'edge': simulated_region(),
            'cloud': {'mode': 'always_succeeds'},
        }, invocation={'entry_region': 'cloud'})
        stats = Simulation(scenario).run(arrivals=1).metrics
        self.assertEqual((stats['total']['arrivals'], stats['total']['completed']), (1, 1))
        self.assertEqual(stats['total']['mean_response_time_s'], 0)
        self.assertEqual(stats['regions']['cloud']['external_arrivals'], 1)
        self.assertEqual(stats['regions']['edge']['external_arrivals'], 0)
        self.assertEqual(stats['regional_offloads'], [])

    def test_routing_and_edge_neighbors_stay_in_their_region(self):
        scenario = Scenario([dict(FUNCTION, first_arrival_time=2)], scheduler={'nearby_interval': 1}, regions={
            'edge': simulated_region(policy='edgeonly'),
            'other': simulated_region(),
        }, invocation={'can_offload': True})
        stats = Simulation(scenario).run(arrivals=1).metrics
        self.assertEqual(stats['regions']['edge']['invocations']['rejected'], 1)
        self.assertEqual(stats['regions']['other']['invocations']['arrivals'], 0)

    def test_invalid_region_references_are_rejected_at_load_time(self):
        for options, field in (
            ({'primary_region': 'missing'}, 'primary_region'),
            ({'invocation': {'entry_region': 'missing'}}, 'entry_region'),
            ({'regions': {'edge': simulated_region(remote_region='missing')}}, 'remote_region'),
            ({'regions': {'edge': simulated_region(remote_region='edge')}}, 'remote_region'),
            ({'infrastructure': {'rings': []}}, 'infrastructure'),
            ({'regions': {'edge': {'mode': 'always_succeeds', 'offload_latency_s': -1}}}, 'offload_latency_s'),
            ({'regions': {'edge': {'offload_latency_by_source_s': {'missing': 1}}}}, 'offload_latency_by_source_s'),
            ({'regions': {'edge': {'offload_latency_by_source_s': {'edge': 1}}}}, 'offload_latency_by_source_s'),
            ({'regions': {'edge': {'offload_latency_by_source_s': {'other': -1}}, 'other': {}}},
             'offload_latency_by_source_s'),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                Scenario([FUNCTION], **({'regions': {'edge': simulated_region()}} | options))
