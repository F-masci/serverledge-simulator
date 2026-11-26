"""Ingressi regionali e trasferimenti tra infrastrutture indipendenti."""

import unittest

from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario
from simulator.model.execution.request import RequestStatus
from simulator.observer import NotificationType, observer

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
            'nodes': [{'name': 'worker', **node_settings}],
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
        simulation = Simulation(Scenario(
            [dict(FUNCTION, arrival_rate=1)], invocation={'can_offload': True}, regions={
                'edge': simulated_region(policy='cloudonly', remote_region='cloud'),
                'cloud': {
                    'mode': 'always_succeeds', 'service_time_s': 0.375,
                    'offload_latency_s': 0.5, 'offload_latency_by_source_s': {'edge': 0.25},
                },
            },
        ))

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

        registered, arrived, forwarded, delivered, admitted, completed = self.notifications
        self.assertEqual([event.type for event in self.notifications], [
            NotificationType.NODE_REGISTERED, NotificationType.REQUEST_ARRIVED,
            NotificationType.REQUEST_FORWARDED, NotificationType.REQUEST_DELIVERED,
            NotificationType.REQUEST_ACCEPTED, NotificationType.REQUEST_COMPLETED,
        ])
        edge, cloud = simulation.regions['edge'], simulation.regions['cloud']
        self.assertIs(registered.region, edge)
        self.assertEqual((forwarded.source_region, forwarded.region, delivered.sim_time_s), (edge, cloud, 0.125))
        self.assertEqual({event.request for event in (forwarded, delivered, admitted, completed)}, {arrived.request})
        self.assertEqual((arrived.region, completed.region), (edge, cloud))
        self.assertIsNone(completed.node)
        self.assertIsNone(completed.container)

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
        for source, admission, completion in (
            ('north', 1, 2.5), ('local', 0, 0.5), ('south', 2, 4.5),
        ):
            with self.subTest(source=source):
                simulation = Simulation(Scenario(
                    [FUNCTION], regions=regions,
                    invocation={'entry_region': source, 'can_offload': True},
                ))
                simulation.run(arrivals=1)
                self.advance_to(simulation, completion)
                received, = [event for event in self.notifications
                             if event.run_id == simulation.run_id
                             and event.type is NotificationType.REQUEST_ACCEPTED]
                completed, = [event for event in self.notifications
                              if event.run_id == simulation.run_id
                              and event.type is NotificationType.REQUEST_COMPLETED]
                self.assertEqual(received.sim_time_s, admission)
                self.assertEqual(completed.response_time_s, completion)

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
        self.assertIs(request._entry_region, simulation.regions['edge'])
        self.assertIs(request.region, simulation.regions['relay'])
        self.assertIs(request.infrastructure, simulation.regions['relay'].infrastructure)
        self.assertIs(request.ring, request.infrastructure.rings['amd64'])
        payload = request.to_dict()
        self.assertEqual((payload['entry_region'], payload['region']), ('edge', 'relay'))
        self.assertEqual(payload['ring_id'], 'amd64')
        self.assertEqual(stats['total']['rejected'], 1)
        self.assertEqual(stats['regional_offloads'], [{'source': 'edge', 'destination': 'relay', 'count': 1}])

    def test_remote_rejection_pays_return_latency(self):
        scenario = Scenario([FUNCTION], invocation={'can_offload': True}, regions={
            'edge': simulated_region(policy='cloudonly', remote_region='relay'),
            'relay': simulated_region(latency=4, policy='cloudonly', remote_region='cloud'),
            'cloud': {'mode': 'always_succeeds'},
        })
        simulation = Simulation(scenario)
        simulation.run(arrivals=1)

        # Alla consegna (t=2) il rifiuto entra in fase di ritorno; in_transit/pending restano 1.
        self.advance_to(simulation, 2)
        mid = simulation.metrics_snapshot()['total']
        self.assertEqual((mid['arrivals'], mid['rejected'], mid['in_transit'], mid['pending']), (1, 0, 1, 1))

        self.advance_to(simulation, 4)
        final = simulation.metrics_snapshot()['total']
        self.assertEqual((final['arrivals'], final['rejected'], final['in_transit'], final['pending']), (1, 1, 0, 0))
        rejected, = [
            event for event in self.notifications
            if event.run_id == simulation.run_id and event.type is NotificationType.REQUEST_REJECTED
        ]
        self.assertEqual(rejected.sim_time_s, 4)

    def test_remote_region_uses_the_balancer_or_checks_the_registered_node(self):
        cloud = simulated_region()
        cloud['infrastructure']['rings'].insert(0, {
            'id': 'arm', 'arch': 'arm64', 'machine_tag': 'a', 'memory_mb': 128, 'cpus': 1,
            'nodes': [{'name': 'first'}],
        })
        function = dict(FUNCTION, supported_archs=['amd64'])
        for enabled in (False, True):
            with self.subTest(has_load_balancer=enabled):
                scenario = Scenario([function], invocation={'can_offload': True}, regions={
                    'edge': simulated_region(policy='cloudonly', remote_region='cloud'),
                    'cloud': dict(cloud, has_load_balancer=enabled),
                })
                stats = Simulation(scenario).run(arrivals=1).metrics
                self.assertEqual(stats['regions']['cloud']['invocations']['accepted'], int(enabled))
                self.assertEqual(stats['regions']['edge']['invocations']['rejected'], int(not enabled))
                self.assertEqual(len(stats['regional_offloads']), int(enabled))

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
        scenario = Scenario([dict(FUNCTION, first_arrival_time=2)], regions={
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
            kwargs = {'regions': {'edge': simulated_region()}} | options
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                Scenario([FUNCTION], **kwargs)
