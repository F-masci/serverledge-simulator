"""Contratto di composizione dell'infrastruttura del simulatore.

Infrastructure e ArchitectureRing definiscono appartenenza e creazione delegata;
non rappresentano ancora una politica di hashing o una replica del bilanciatore.
L'identità dei nodi usa i campi descritti da internal/node/node.go, NodeID,
nel repository Infrascaler. I riferimenti vengono mantenuti senza copiare lo stato.
"""

import unittest
from unittest.mock import patch

from sim.model.infrastructure import Infrastructure
from sim.model.ring import ArchitectureRing
from sim.model.function import Function, BuiltinRuntime


class InfrastructureTest(unittest.TestCase):
    def test_creation_is_delegated_to_architecture_ring(self):
        """L'infrastruttura inoltra la creazione all'anello esistente senza sostituirlo."""
        infra = Infrastructure()
        ring = infra.create_ring('x86-small', arch='amd64', memory=512, cpus=2)
        with patch.object(infra, 'create_ring', wraps=infra.create_ring) as create_ring:
            with patch.object(ring, 'create_node', wraps=ring.create_node) as create:
                node = infra.create_node('x86-small', 'n1', 'edge', 'cpu')
        create_ring.assert_called_once_with('x86-small')
        create.assert_called_once_with('n1', 'edge', 'cpu', idle_timeout=600)
        self.assertIs(infra.rings['x86-small'], ring)
        self.assertIs(infra.create_ring('x86-small', arch='amd64', memory=512, cpus=2), ring)
        self.assertIs(ring.nodes[0], node)
        self.assertEqual(node.id.arch, ring.arch)
        self.assertEqual(node.id.machine_tag, 'cpu')

    def test_rings_keep_independent_live_node_references(self):
        """Anelli distinti mantengono oggetti vivi; le tuple non copiano i nodi."""
        infra = Infrastructure()
        infra.create_ring('x86-small', arch='amd64', memory=512, cpus=2)
        infra.create_ring('arm-large', arch='arm64', memory=1024, cpus=4)
        a = infra.create_node('x86-small', 'x86', 'edge')
        b = infra.create_node('arm-large', 'arm', 'edge')
        ring = infra.rings['x86-small']
        snapshot = ring.nodes
        fun = Function('f', 128, 1, BuiltinRuntime('python314'))
        a.acquire(fun)
        self.assertEqual(snapshot[0].resources.used_cpus, 1)
        self.assertEqual(b.resources.used_cpus, 0)
        self.assertIn(a, ring)
        self.assertNotIn(b, ring)
        self.assertEqual(tuple(ring), (a,))
        infra.create_node('x86-small', 'x86-2', 'edge')
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(ring), 2)
        with self.assertRaises(TypeError):
            infra.rings['other'] = ring
        with self.assertRaises(AttributeError):
            ring.nodes = ()

    def test_ring_capacities_are_shared_but_reservations_are_independent(self):
        """Stessa architettura può avere ring di taglie diverse; ogni ring resta omogeneo."""
        infra = Infrastructure()
        small = infra.create_ring('small', arch='amd64', memory=512, cpus=2)
        large = infra.create_ring('large', arch='amd64', memory=1024, cpus=4)
        a = infra.create_node('small', 'a', 'edge')
        b = infra.create_node('small', 'b', 'edge')
        c = infra.create_node('large', 'c', 'edge')
        for node in (a, b):
            self.assertEqual((node.resources.total_memory, node.resources.total_cpus), (512, 2))
        self.assertEqual(c.resources.total_memory, 1024)
        a.acquire(Function('f', 128, 1, BuiltinRuntime('python314')))
        self.assertEqual(a.resources.used_cpus, 1)
        self.assertEqual(b.resources.used_cpus, 0)
        self.assertEqual(small.cpus, 2)
        self.assertIsNot(small, large)

    def test_conflicts_and_invalid_creation_preserve_rings(self):
        """Duplicati e riconfigurazioni falliscono senza cambiare inventario o capacità."""
        infra = Infrastructure()
        ring = infra.create_ring('small', arch='amd64', memory=512, cpus=2)
        node = infra.create_node('small', 'n1', 'edge')
        with self.assertRaises(ValueError):
            infra.create_node('small', 'n1', 'edge')
        with self.assertRaises(ValueError):
            infra.create_ring('small', cpus=4)
        with self.assertRaises(ValueError):
            infra.create_node('missing', 'n2', 'edge')
        for memory in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                infra.create_ring('bad', arch='amd64', memory=memory, cpus=2)
        self.assertEqual(tuple(infra.rings), ('small',))
        self.assertEqual(ring.nodes, (node,))
        self.assertEqual(ring.cpus, 2)
        with self.assertRaises(AttributeError):
            ring.memory = 1024


if __name__ == '__main__':
    unittest.main()
