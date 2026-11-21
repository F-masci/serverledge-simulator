"""Scelte di scheduling delle invocazioni verificate contro Infrascaler."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sim.core.simulation import Simulation
from sim.io.scenario import Scenario
from sim.model.function import BuiltinRuntime, Function
from sim.model.infrastructure import Infrastructure
from sim.model.offloading import NodeScheduling, OffloadingScheduler, SchedulingPolicy
from sim.model.region import Region
from sim.model.request import Request, RequestStatus


class OffloadingTest(unittest.TestCase):
    def setUp(self):
        self.infrastructure = Infrastructure()
        self.function = Function('f', 128, 1, BuiltinRuntime('python314'), supported_archs=('amd64',))

    def node(self, name, *, arch='amd64', tag=None, memory=128, cpus=1, queue_capacity=0):
        self.infrastructure.create_ring(name, arch=arch, memory=memory, cpus=cpus)
        return self.infrastructure.create_node(name, name, tag, queue_capacity=queue_capacity)

    def test_default_policy_queues_locally_even_if_offloading_is_allowed(self):
        origin = self.node('origin', queue_capacity=1)
        active, waiting = (Request(self.function, can_offload=True) for _ in range(2))
        self.assertIs(self.infrastructure.submit(active, node=origin), origin)
        origin.creation_succeeded(active.container, 0)
        self.assertIs(self.infrastructure.submit(waiting, node=origin), origin)
        self.assertIs(waiting.status, RequestStatus.QUEUED)
        self.assertIsNone(waiting.to_dict()['offloaded_from_node_id'])
        active.complete(1)
        self.assertIs(waiting.status, RequestStatus.ACCEPTED)
        self.assertIs(waiting.container, active.container)

    def test_edgeonly_falls_back_immediately_when_no_neighbor_is_suitable(self):
        origin = self.node('origin')
        peer = self.node('arm', arch='arm64')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(peer,),
        ))

        fallback = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(fallback, node=origin), origin)
        self.assertIsNone(fallback.to_dict()['offloaded_from_node_id'])

    def test_edgeonly_local_fallback_does_not_use_the_fifo(self):
        origin = self.node('origin', queue_capacity=1)
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True,
        ))

        active = Request(self.function)
        fallback = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(active, node=origin), origin)
        self.assertIsNone(self.infrastructure.submit(fallback, node=origin))
        self.assertIs(fallback.status, RequestStatus.REJECTED)
        self.assertIs(fallback.node, origin)

    def test_edgeonly_uses_current_memory_for_each_invocation(self):
        origin = self.node('origin', queue_capacity=1)
        small = self.node('small', memory=256)
        large = self.node('large', memory=512, cpus=2)
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(small, large),
        ))

        first = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(first, node=origin, now=20), large)
        self.assertEqual(first.to_dict()['offloaded_from_node_id'], origin.id.key)
        self.assertFalse(first.can_offload)
        blocker = Function('blocker', 384, 1, BuiltinRuntime('python314'))
        self.assertIs(self.infrastructure.submit(Request(blocker), node=large, now=21), large)

        second = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(second, node=origin, now=21), small)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_edgeonly_forwards_to_positive_memory_even_if_insufficient(self):
        origin = self.node('origin')
        peer = self.node('peer', memory=64)
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(peer,),
        ))

        request = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(request, node=origin, now=20))
        self.assertIs(request.node, peer)
        self.assertEqual(request.to_dict()['offloaded_from_node_id'], origin.id.key)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_edgecloud_runs_locally_first_then_offloads_to_configured_region(self):
        edge = self.node('edge', queue_capacity=1)
        cloud = Region('cloud')
        self.infrastructure.configure_scheduler(edge, NodeScheduling(
            SchedulingPolicy.EDGE_CLOUD, remote_region=cloud,
        ))
        local = Request(self.function, can_offload=True)
        forbidden = Request(self.function)
        self.assertIs(self.infrastructure.submit(local, node=edge), edge)
        self.assertIsNone(self.infrastructure.submit(forbidden, node=edge))
        self.assertIs(forbidden.status, RequestStatus.REJECTED)
        # La policy sceglie la singola regione remota quando il nodo è saturo.
        scheduler = OffloadingScheduler.create(
            lambda: (edge,), NodeScheduling(SchedulingPolicy.EDGE_CLOUD, remote_region=cloud),
        )
        remote = Request(self.function, can_offload=True)
        self.assertIs(scheduler.schedule(remote, edge), cloud)
        self.assertIs(remote.status, RequestStatus.CREATED)

    def test_cloudonly_rejects_without_permission_or_remote_region(self):
        origin = self.node('origin')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(SchedulingPolicy.CLOUD_ONLY))
        for can_offload in (False, True):
            with self.subTest(can_offload=can_offload):
                request = Request(self.function, can_offload=can_offload)
                self.assertIsNone(self.infrastructure.submit(request, node=origin))
                self.assertIs(request.status, RequestStatus.REJECTED)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_local_rejection_after_proxy_selection_does_not_retry_another_node(self):
        ring = self.infrastructure.create_ring('edge', arch='amd64', memory=128, cpus=1)
        for name in ('one', 'two'):
            self.infrastructure.create_node('edge', name)
        primary = ring.select_node(self.function)
        self.infrastructure.configure_scheduler(primary, NodeScheduling(SchedulingPolicy.CLOUD_ONLY))

        request = Request(self.function)
        self.assertIsNone(self.infrastructure.submit(request))
        self.assertIs(request.node, primary)
        self.assertIs(request.status, RequestStatus.REJECTED)
        self.assertTrue(all(node.resources.used_cpus == 0 for node in ring))

    def test_scenario_offloads_the_first_invocation_without_waiting_for_discovery(self):
        scenario_data = """
[scheduler]
policy = "edgeonly"

[invocation]
entry_node = "edge-1"
can_offload = true

[[functions]]
name = "f"
runtime = "python314"
memory_mb = 128
cpu_demand = 1
arrival_rate = 1
service_rate = 1
first_arrival_time = 0
arrival_distribution = "deterministic"
service_distribution = "deterministic"

[[infrastructure.rings]]
id = "amd64"
arch = "amd64"
memory_mb = 256
cpus = 1

[[infrastructure.rings.nodes]]
name = "edge-1"
nearest_neighbors = ["edge-2"]

[[infrastructure.rings.nodes]]
name = "edge-2"
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'offloading.toml'
            path.write_text(scenario_data, encoding='utf-8')
            simulation = Simulation(Scenario.load(path))
        requests = []
        infrastructure = simulation.regions['default'].infrastructure
        infrastructure.on_request_submitted = lambda _, request, node: requests.append(request)

        statistics = simulation.run(arrivals=1)
        accepted, = requests
        self.assertIs(accepted.status, RequestStatus.ACCEPTED)
        self.assertEqual(accepted.node.id.key, 'edge-2')
        self.assertEqual(accepted.to_dict()['offloaded_from_node_id'], 'edge-1')
        self.assertEqual(statistics.metrics['total']['arrivals'], 1)
