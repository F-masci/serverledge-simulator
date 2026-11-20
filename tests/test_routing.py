"""Percorso di inoltro iniziale e compatibilità dell'hashing.

Riferimenti Infrascaler: internal/lb/hashRing.go, hash, HashRing.Add e
HashRing.Get per l'ordine delle repliche virtuali e la ricerca di altri nodi;
internal/lb/generalLoadBalancer.go per il filtro di compatibilità.
Il simulatore usa l'ammissione locale per provare i candidati del ring scelto.
Infrascaler preseleziona i nodi tramite metriche e può ripiegare su altri pool.
"""

import unittest

from sim.model.infrastructure import Infrastructure
from sim.model.request import Request, RequestStatus
from sim.model.function import Function, BuiltinRuntime


def function(name='hello', archs=('amd64', 'arm64')):
    return Function(name, 128, 1, BuiltinRuntime('python314'), supported_archs=archs)


class RoutingTest(unittest.TestCase):
    def test_empty_and_incompatible_rings_are_skipped(self):
        """Non si inoltra a ring vuoti o non compatibili; senza destinazione restituisce None."""
        infra = Infrastructure()
        self.assertIsNone(infra.submit(Request(function())))
        infra.create_ring('empty', arch='amd64', memory=512, cpus=2)
        infra.create_ring('arm', arch='arm64', memory=512, cpus=2)
        arm = infra.create_node('arm', 'a', 'edge')
        self.assertIsNone(infra.submit(Request(function(archs=('amd64',)))))
        self.assertEqual(arm.resources.used_cpus, 0)
        request = Request(function())
        self.assertIs(infra.submit(request), arm)
        self.assertIs(request.node, arm)

    def test_hash_lookup_matches_reference_at_boundary_and_wraparound(self):
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2, replicas=2)
        first = infra.create_node('r', 'n1', 'edge')
        second = infra.create_node('r', 'n2', 'edge')

        # Go HashRing.Add/Get: n2#1 è l'ultimo punto di n2, seguito da n1.
        # L'hash di foobar supera tutti i punti e riparte dal primo (n2).
        for name, expected in (('n2#1', second), ('hello', first), ('foobar', second)):
            with self.subTest(function=name):
                self.assertIs(ring.select_node(function(name)), expected)

    def test_same_function_uses_reference_order_until_all_nodes_are_saturated(self):
        infra = Infrastructure()
        infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        nodes = {name: infra.create_node('r', name, 'edge') for name in ('one', 'two', 'three')}

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
        self.assertIs(rejected.node, nodes['one'])
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(rejected.rejection_reason, 'out_of_resources')

    def test_removing_primary_updates_routing_before_notification(self):
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        primary = infra.create_node('r', 'one', 'edge')
        secondary = infra.create_node('r', 'three', 'edge')
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

        replacement = infra.create_node('r', 'one', 'edge')
        self.assertIsNot(replacement, primary)
        self.assertIs(infra.submit(Request(function())), replacement)


if __name__ == '__main__':
    unittest.main()
