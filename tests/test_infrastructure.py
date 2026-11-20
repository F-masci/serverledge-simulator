"""Inventario e isolamento delle risorse dei nodi simulati."""

import unittest

from sim.model.infrastructure import Infrastructure
from sim.model.function import Function, BuiltinRuntime


class InfrastructureTest(unittest.TestCase):
    def test_creation_callbacks_observe_registered_ring_and_node(self):
        infra = Infrastructure()
        created_rings, created_nodes = [], []

        def ring_created(infrastructure, ring):
            self.assertIs(infrastructure.get_ring(ring.id), ring)
            created_rings.append(ring)

        def node_created(infrastructure, ring, node):
            self.assertIs(infrastructure.get_ring(ring.id), ring)
            self.assertIn(node, ring)
            created_nodes.append(node)

        infra.on_ring_created = ring_created
        infra.on_node_created = node_created
        ring = infra.create_ring('x86-small', arch='amd64', memory=512, cpus=2)
        node = infra.create_node('x86-small', 'n1', 'cpu')

        self.assertIs(infra.create_ring('x86-small', cpus=4), ring)
        self.assertEqual(ring.cpus, 2)
        self.assertEqual(created_rings, [ring])
        self.assertEqual(created_nodes, [node])
        self.assertEqual((node.id.arch, node.id.machine_tag), ('amd64', 'cpu'))

    def test_ring_capacities_are_shared_but_reservations_are_independent(self):
        """Stessa architettura può avere ring di taglie diverse; ogni ring resta omogeneo."""
        infra = Infrastructure()
        small = infra.create_ring('small', arch='amd64', memory=512, cpus=2)
        large = infra.create_ring('large', arch='amd64', memory=1024, cpus=4)
        a = infra.create_node('small', 'a')
        b = infra.create_node('small', 'b')
        c = infra.create_node('large', 'c')
        for node in (a, b):
            self.assertEqual((node.resources.total_memory, node.resources.total_cpus), (512, 2))
        self.assertEqual((c.resources.total_memory, c.resources.total_cpus), (1024, 4))
        a.acquire(Function('f', 128, 1, BuiltinRuntime('python314')))
        self.assertEqual(a.resources.used_cpus, 1)
        self.assertEqual(b.resources.used_cpus, 0)
        self.assertEqual(small.cpus, 2)
        self.assertIsNot(small, large)

    def test_conflicts_and_invalid_creation_preserve_rings(self):
        """Creazioni non valide lasciano intatti inventario e capacità."""
        infra = Infrastructure()
        ring = infra.create_ring('small', arch='amd64', memory=512, cpus=2)
        node = infra.create_node('small', 'n1')
        with self.assertRaises(ValueError):
            infra.create_node('small', 'n1')
        with self.assertRaises(ValueError):
            infra.create_node('missing', 'n2')
        for memory in (0, -1, 1.5, True):
            with self.subTest(memory=memory):
                with self.assertRaises(ValueError):
                    infra.create_ring('bad', arch='amd64', memory=memory, cpus=2)
        self.assertEqual(tuple(infra.rings), ('small',))
        self.assertEqual(ring.nodes, (node,))
        self.assertEqual(ring.cpus, 2)


if __name__ == '__main__':
    unittest.main()
