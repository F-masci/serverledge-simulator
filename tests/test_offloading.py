"""Scelte di scheduling delle invocazioni verificate contro Infrascaler."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from sim.core.simulation import Simulation
from sim.io.scenario import Scenario
from sim.model.function import BuiltinRuntime, Function
from sim.model.infrastructure import Infrastructure
from sim.model.offloading import NodeScheduling, SchedulingPolicy
from sim.model.request import Request, RequestStatus


class OffloadingTest(unittest.TestCase):
    def setUp(self):
        self.infrastructure = Infrastructure()
        self.function = Function('f', 128, 1, BuiltinRuntime('python314'), supported_archs=('amd64',))

    def node(self, name, *, area='edge', arch='amd64', tag=None, memory=128, cpus=1, queue_capacity=0):
        self.infrastructure.create_ring(name, arch=arch, memory=memory, cpus=cpus)
        return self.infrastructure.create_node(name, name, area, tag, queue_capacity=queue_capacity)

    def test_default_policy_queues_locally_even_if_offloading_is_allowed(self):
        origin = self.node('origin', queue_capacity=1)
        active, waiting = (Request(self.function, can_offload=True) for _ in range(2))
        self.assertIs(self.infrastructure.submit(active, node=origin), origin)
        origin.creation_succeeded(active.container, 0)
        self.assertIs(self.infrastructure.submit(waiting, node=origin), origin)
        self.assertIs(waiting.status, RequestStatus.QUEUED)
        self.assertIsNone(waiting.offloaded_from)
        active.complete(1)
        self.assertIs(waiting.status, RequestStatus.ACCEPTED)
        self.assertIs(waiting.container, active.container)

    def test_edgeonly_falls_back_only_after_monitoring_returns_no_suitable_node(self):
        origin = self.node('origin')
        peer = self.node('arm', arch='arm64')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(peer,),
        ))

        too_early = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(too_early, node=origin, now=19))
        self.assertIs(too_early.status, RequestStatus.REJECTED)

        self.infrastructure.refresh_neighbors()
        fallback = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(fallback, node=origin, now=20), origin)
        self.assertIsNone(fallback.offloaded_from)

    def test_edgeonly_selects_maximum_reported_memory_and_honors_cache_expiry(self):
        origin = self.node('origin', queue_capacity=1)
        small = self.node('small', memory=256)
        large = self.node('large', memory=512, cpus=2)
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(small, large),
        ))
        self.infrastructure.refresh_neighbors()

        first = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(first, node=origin, now=20), large)
        self.assertIs(first.offloaded_from, origin)
        self.assertFalse(first.can_offload)
        blocker = Function('blocker', 384, 1, BuiltinRuntime('python314'))
        self.assertIs(self.infrastructure.submit(Request(blocker), node=large, now=21), large)
        self.infrastructure.refresh_neighbors()

        cached = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(cached, node=origin, now=79))
        self.assertIs(cached.status, RequestStatus.REJECTED)
        self.assertIs(cached.node, large)
        self.assertIs(cached.offloaded_from, origin)
        self.assertEqual(origin.resources.used_cpus, 0)

        expired = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(expired, node=origin, now=80), small)

    def test_edgeonly_forwards_to_positive_memory_even_if_insufficient(self):
        origin = self.node('origin')
        peer = self.node('peer', memory=64)
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(peer,),
        ))
        self.infrastructure.refresh_neighbors()

        request = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(request, node=origin, now=20))
        self.assertIs(request.node, peer)
        self.assertIs(request.offloaded_from, origin)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_edgecloud_runs_locally_first_then_uses_the_registered_cloud_node(self):
        edge = self.node('edge', queue_capacity=1)
        cloud = self.node('cloud', area='cloud')
        self.infrastructure.configure_scheduler(edge, NodeScheduling(SchedulingPolicy.EDGE_CLOUD, remote_area='cloud'))
        self.infrastructure.refresh_remote_targets()

        local = Request(self.function, can_offload=True)
        forbidden = Request(self.function)
        remote = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(local, node=edge), edge)
        self.assertIsNone(self.infrastructure.submit(forbidden, node=edge))
        self.assertIs(forbidden.status, RequestStatus.REJECTED)
        self.assertIs(self.infrastructure.submit(remote, node=edge), cloud)
        self.assertIs(remote.offloaded_from, edge)

    def test_cloudonly_rejects_without_permission_or_compatible_registered_target(self):
        origin = self.node('origin')
        wrong_arch = self.node('a-cloud', area='cloud', arch='arm64', tag='a')
        other = self.node('b-cloud', area='cloud', tag='b')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(SchedulingPolicy.CLOUD_ONLY, remote_area='cloud'))
        self.infrastructure.refresh_remote_targets()

        for can_offload in (False, True):
            with self.subTest(can_offload=can_offload):
                request = Request(self.function, can_offload=can_offload)
                self.assertIsNone(self.infrastructure.submit(request, node=origin))
                self.assertIs(request.status, RequestStatus.REJECTED)
                self.assertIsNone(request.offloaded_from)
        for node in (origin, wrong_arch, other):
            self.assertEqual(node.resources.used_cpus, 0)

    def test_cloud_load_balancer_is_preferred_and_cannot_offload_again(self):
        self.infrastructure = Infrastructure(load_balancer_areas=('cloud',))
        origin = self.node('origin', arch='arm64')
        self.node('a-cloud', area='cloud', arch='arm64', tag='a')
        receiver = self.node('b-cloud', area='cloud', tag='b')
        other = self.node('far', area='other')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(SchedulingPolicy.EDGE_CLOUD, remote_area='cloud'))
        self.infrastructure.refresh_remote_targets()

        request = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(request, node=origin), receiver)
        self.assertIs(request.offloaded_from, origin)

        saturated = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(saturated, node=origin))
        self.assertIs(saturated.offloaded_from, origin)
        self.assertIsNone(saturated.node)
        self.assertEqual(other.resources.used_cpus, 0)

        self.infrastructure.configure_scheduler(
            receiver, NodeScheduling(SchedulingPolicy.CLOUD_ONLY, remote_area='other'),
        )
        self.infrastructure.refresh_remote_targets()
        receiver.creation_succeeded(request.container, 0)
        request.complete(1)

        forwarded = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(forwarded, node=origin, now=1))
        self.assertIs(forwarded.node, receiver)
        self.assertIs(forwarded.offloaded_from, origin)
        self.assertEqual(other.resources.used_cpus, 0)

    def test_local_rejection_after_proxy_selection_does_not_retry_another_node(self):
        ring = self.infrastructure.create_ring('edge', arch='amd64', memory=128, cpus=1)
        for name in ('one', 'two'):
            self.infrastructure.create_node('edge', name, 'edge')
        primary = ring.select_node(self.function)
        self.infrastructure.configure_scheduler(primary, NodeScheduling(SchedulingPolicy.CLOUD_ONLY))

        request = Request(self.function)
        self.assertIsNone(self.infrastructure.submit(request))
        self.assertIs(request.node, primary)
        self.assertIs(request.status, RequestStatus.REJECTED)
        self.assertTrue(all(node.resources.used_cpus == 0 for node in ring))

    def test_scenario_delivers_direct_invocations_after_neighbor_monitoring(self):
        scenario_data = """
[scheduler]
policy = "edgeonly"

[invocation]
entry_node = "edge-1"
entry_area = "edge"
can_offload = true

[[functions]]
name = "f"
runtime = "python314"
memory_mb = 128
cpu_demand = 1
arrival_rate = 1
service_rate = 1
first_arrival_time = 19
arrival_distribution = "deterministic"
service_distribution = "deterministic"

[[infrastructure.rings]]
id = "amd64"
arch = "amd64"
memory_mb = 256
cpus = 1

[[infrastructure.rings.nodes]]
name = "edge-1"
area = "edge"
nearest_neighbors = ["edge-2"]

[[infrastructure.rings.nodes]]
name = "edge-2"
area = "edge"
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'offloading.toml'
            path.write_text(scenario_data, encoding='utf-8')
            simulation = Simulation(Scenario.load(path))
        requests = []
        simulation._ctx.infrastructure.on_request_submitted = lambda _, request, node: requests.append(request)

        statistics = simulation.run(arrivals=2)
        rejected, accepted = requests
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertIs(accepted.status, RequestStatus.ACCEPTED)
        self.assertEqual(accepted.node.id.key, 'edge-2')
        self.assertEqual(accepted.to_dict()['offloaded_from_node_id'], 'edge-1')
        self.assertEqual(statistics.metrics['total']['arrivals'], 2)
