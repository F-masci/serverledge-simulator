"""Percorso di inoltro iniziale e compatibilità dell'hashing.

Riferimenti Infrascaler: internal/lb/hashRing.go, hash, HashRing.Add e
HashRing.Get per l'ordine delle repliche virtuali e la ricerca di altri nodi;
internal/lb/generalLoadBalancer.go per il filtro di compatibilità e il fallback.
"""

from dataclasses import replace
import unittest

from sim.model.infrastructure import Infrastructure
from sim.model.load_balancer import (
    ArchitectureUnawareLoadBalancer, MABLoadBalancer, RandomLoadBalancer, RoundRobinLoadBalancer,
)
from sim.model.request import Request, RequestStatus
from sim.model.function import Function, BuiltinRuntime


def function(name='hello', archs=('amd64', 'arm64')):
    return Function(name, 128, 1, BuiltinRuntime('python314'), supported_archs=archs)


class RoutingTest(unittest.TestCase):
    def test_random_balancer_draws_only_compatible_rings_with_reproducible_seed(self):
        infra = Infrastructure()
        for ring_id, arch, tag in (
            ('empty', 'amd64', None),
            ('plain', 'amd64', 'x86/tiny'),
            ('wrong-arch', 'arm64', 'arm/gpu/nvidia'),
            ('first', 'amd64', 'x86-large/gpu/nvidia'),
            ('second', 'amd64', 'x86-tiny/gpu/nvidia'),
        ):
            infra.create_ring(ring_id, arch=arch, memory=512, cpus=2)
            if tag is not None:
                infra.create_node(ring_id, ring_id, tag)
        constrained = replace(function(archs=('amd64',)), tag_pattern='GPU, nvidia')

        def draws(seed):
            balancer = RandomLoadBalancer(seed=seed)
            return [
                balancer.submit(Request(constrained), infra.rings.values()).id
                for _ in range(32)
            ]

        baseline = draws(42)
        self.assertEqual(set(baseline), {'first', 'second'})
        self.assertEqual(baseline, draws(42))
        self.assertNotEqual(baseline, draws(43))

    def test_machine_tag_requirements_match_reference_patterns(self):
        tag = 'x86-large/gpu/nvidia'
        for pattern, expected in (
            ('', True), ('x86', True), ('large', True), ('x86-large/gpu', True),
            ('*/gpu/nvidia', True), ('^x86-.*/gpu.*$', True),
            ('gpu, amd', False), ('x86-tiny/gpu', False), ('^[', False),
        ):
            with self.subTest(pattern=pattern):
                self.assertEqual(replace(function(), tag_pattern=pattern).supports_machine_tag(tag), expected)

    def test_fallback_uses_registration_order_and_rejects_after_saturation(self):
        seed = 42
        infra = Infrastructure(RandomLoadBalancer(seed=seed))
        nodes = {}
        for ring_id in ('gpu-first', 'gpu-second', 'gpu-third', 'plain'):
            infra.create_ring(ring_id, arch='amd64', memory=128, cpus=1)
            nodes[ring_id] = infra.create_node(
                ring_id, ring_id, f'x86/{ring_id}', queue_capacity=1,
            )

        fun = replace(function(), tag_pattern='x86/gpu-*')
        requests = (Request(fun), Request(fun))
        predictor = RandomLoadBalancer(seed=seed)
        order = predictor.candidate_rings(requests[0], infra.rings.values())
        primary, fallback, last = (nodes[ring.id] for ring in order)
        fallback_order = [ring_id for ring_id in nodes if ring_id not in (order[0].id, 'plain')]
        self.assertEqual([ring.id for ring in order[1:]], fallback_order)
        self.assertTrue(primary.submit(Request(fun)))

        self.assertIs(infra.submit(requests[0]), fallback)
        self.assertIs(requests[0].status, RequestStatus.ACCEPTED)
        self.assertTrue(last.submit(Request(fun)))

        rejected = requests[1]
        self.assertIsNone(infra.submit(rejected))
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(rejected.rejection_reason, 'out_of_resources')
        payload = rejected.to_dict()
        self.assertIsNone(payload['node_id'])
        self.assertIsNone(payload['container_id'])
        self.assertEqual(nodes['plain'].resources.used_cpus, 0)

    def test_ring_with_mixed_node_tags_only_admits_compatible_nodes(self):
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        regular = infra.create_node('r', 'one')
        gpu = infra.create_node('r', 'three', 'x86/gpu')
        self.assertIs(ring.select_node(function()), regular)

        constrained = replace(function(), tag_pattern='gpu')
        self.assertIs(infra.submit(Request(constrained)), gpu)
        rejected = Request(replace(function(), tag_pattern='arm*'))
        self.assertIsNone(infra.submit(rejected))
        self.assertEqual(rejected.rejection_reason, 'no_destination')
        self.assertEqual(regular.resources.used_cpus, 0)

    def test_round_robin_balancer_cycles_compatible_rings_in_registration_order(self):
        infra = Infrastructure()
        for ring_id, arch, tag in (
            ('empty', 'amd64', None),
            ('first', 'amd64', 'x86-large/gpu/nvidia'),
            ('wrong-arch', 'arm64', 'arm/gpu/nvidia'),
            ('second', 'amd64', 'x86-tiny/gpu/nvidia'),
            ('third', 'amd64', 'x86-mid/gpu/nvidia'),
        ):
            infra.create_ring(ring_id, arch=arch, memory=512, cpus=2)
            if tag is not None:
                infra.create_node(ring_id, ring_id, tag)

        balancer = RoundRobinLoadBalancer()
        all_gpu = replace(function(archs=('amd64',)), tag_pattern='gpu')
        only_third = replace(function(archs=('amd64',)), tag_pattern='x86-mid')

        # Cicla sugli anelli compatibili saltando quelli vuoti o con architettura incompatibile.
        draws = [
            balancer.submit(Request(all_gpu), infra.rings.values()).id
            for _ in range(4)
        ]
        self.assertEqual(draws, ['first', 'second', 'third', 'first'])

        # Una richiesta compatibile solo con 'third' avanza l'indice globale oltre 'third',
        # quindi la richiesta successiva riparte da 'first'.
        self.assertEqual(balancer.submit(Request(only_third), infra.rings.values()).id, 'third')
        self.assertEqual(balancer.submit(Request(all_gpu), infra.rings.values()).id, 'first')

    def test_unavailable_balancers_fail_on_instantiation(self):
        for balancer, mode in (
            (MABLoadBalancer, 'MAB'),
            (ArchitectureUnawareLoadBalancer, 'ArchitectureUnaware'),
        ):
            with self.subTest(mode=mode), self.assertRaisesRegex(NotImplementedError, mode):
                balancer()

    def test_empty_and_incompatible_rings_are_skipped(self):
        """Non si inoltra a ring vuoti o non compatibili; senza destinazione restituisce None."""
        infra = Infrastructure()
        self.assertIsNone(infra.submit(Request(function())))
        infra.create_ring('empty', arch='amd64', memory=512, cpus=2)
        infra.create_ring('arm', arch='arm64', memory=512, cpus=2)
        arm = infra.create_node('arm', 'a')
        self.assertIsNone(infra.submit(Request(function(archs=('amd64',)))))
        self.assertEqual(arm.resources.used_cpus, 0)
        request = Request(function())
        self.assertIs(infra.submit(request), arm)
        self.assertIs(request.node, arm)

    def test_hash_lookup_matches_reference_at_boundary_and_wraparound(self):
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2, replicas=2)
        first = infra.create_node('r', 'n1')
        second = infra.create_node('r', 'n2')

        # Go HashRing.Add/Get: n2#1 è l'ultimo punto di n2, seguito da n1.
        # L'hash di foobar supera tutti i punti e riparte dal primo (n2).
        for name, expected in (('n2#1', second), ('hello', first), ('foobar', second)):
            with self.subTest(function=name):
                self.assertIs(ring.select_node(function(name)), expected)

    def test_same_function_uses_reference_order_until_all_nodes_are_saturated(self):
        infra = Infrastructure()
        infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        nodes = {name: infra.create_node('r', name) for name in ('one', 'two', 'three')}

        # Ordine ottenuto con FNV-1a e 128 repliche in Go, saturando un nodo alla volta.
        for name in ('one', 'three', 'two'):
            for _ in range(2):
                request = Request(function())
                self.assertIs(infra.submit(request), nodes[name])
                self.assertIs(request.node, nodes[name])
                self.assertIs(request.status, RequestStatus.ACCEPTED)
            self.assertEqual(nodes[name].resources.used_cpus, 2)
            self.assertEqual(nodes[name].resources.busy_pool_used_mem, 256)

        rejected = Request(function())
        self.assertIsNone(infra.submit(rejected))
        self.assertIsNone(rejected.node)
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(rejected.rejection_reason, 'out_of_resources')

    def test_removing_primary_updates_routing_before_notification(self):
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        primary = infra.create_node('r', 'one')
        secondary = infra.create_node('r', 'three')
        self.assertIs(ring.select_node(function()), primary)
        observed = []

        def node_removed(infrastructure, ring, node):
            self.assertIs(infrastructure.get_ring(ring.id), ring)
            self.assertNotIn(node, ring)
            observed.append((node, ring.select_node(function())))

        infra.on_node_removed = node_removed

        self.assertIs(infra.remove_node('r', 'one'), primary)
        with self.assertRaises(ValueError):
            infra.remove_node('r', 'one')
        self.assertIs(infra.submit(Request(function())), secondary)
        self.assertIs(infra.remove_node('r', 'three'), secondary)
        self.assertEqual(observed, [(primary, secondary), (secondary, None)])

        request = Request(function())
        self.assertIsNone(infra.submit(request))
        self.assertEqual(request.rejection_reason, 'no_destination')

        replacement = infra.create_node('r', 'one')
        self.assertIsNot(replacement, primary)
        self.assertIs(infra.submit(Request(function())), replacement)


if __name__ == '__main__':
    unittest.main()
