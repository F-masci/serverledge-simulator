"""Inventario e isolamento delle risorse dei nodi simulati."""

import json
import unittest

from sim.bindings import NodeBindings, RequestBindings
from sim.model.function import Function
from sim.model.infrastructure import Infrastructure
from sim.model.region import Region, RegionMode
from sim.model.request import Request, RequestStatus
from sim.model.runtime import BuiltinRuntime


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
        self.assertEqual(node.resources.total_cpus, 2)
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
        self.assertEqual(a.resources.total_cpus, 2)
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
        self.assertEqual(tuple(ring), (node,))
        self.assertEqual(node.resources.total_cpus, 2)

    def test_node_bindings_are_propagated_to_dynamically_created_rings_and_nodes(self):
        """Anche anelli e nodi creati a runtime ereditano automaticamente i NodeBindings."""
        registered, state_events = [], []
        region = Region('edge')
        bindings = NodeBindings(
            on_node_registered=lambda node: registered.append((node.id.key, node.region.name, node.ring.id)),
            on_container_state_changed=lambda node, container, prev, curr: state_events.append(
                (node.region.name, node.ring.id, node.id.key, prev, curr)
            ),
        )
        infra = Infrastructure(region=region, node_bindings=bindings)
        infra.create_ring('r1', arch='amd64', memory=512, cpus=2)
        n1 = infra.create_node('r1', 'n1')
        # Simulazione di uno scaler che crea dinamicamente un nuovo anello e un nuovo nodo.
        infra.create_ring('r2', arch='arm64', memory=256, cpus=1)
        n2 = infra.create_node('r2', 'n2')

        self.assertEqual(registered, [('n1', 'edge', 'r1'), ('n2', 'edge', 'r2')])
        fun = Function('f', 128, 1, BuiltinRuntime('python314'))
        n1.acquire(fun)
        n2.acquire(fun)
        self.assertEqual([(e[0], e[1], e[2]) for e in state_events], [('edge', 'r1', 'n1'), ('edge', 'r2', 'n2')])

    def test_request_bindings_and_model_serialization(self):
        """RequestBindings collega il ciclo di vita di Request e i model sono serializzabili in JSON."""
        events = []
        bindings = RequestBindings(
            on_accepted=lambda req: events.append(('accepted', req.id)),
            on_completed=lambda req: events.append(('completed', req.id)),
            on_state_changed=lambda req, prev, curr: events.append(('state', prev, curr)),
        )
        fun = Function('f', 128, 1, BuiltinRuntime('python314'))
        region = Region('cloud')
        self.assertIs(region.mode, RegionMode.ALWAYS_SUCCEEDS)
        self.assertTrue(region.is_always_succeeds)

        request = Request(fun)
        bindings.bind(request)
        region.submit(request, now=0.0)
        request.complete(0.0)

        self.assertIn(('accepted', request.id), events)
        self.assertIn(('completed', request.id), events)
        self.assertIn(('state', RequestStatus.CREATED, RequestStatus.ACCEPTED), events)
        self.assertIn(('state', RequestStatus.ACCEPTED, RequestStatus.COMPLETED), events)

        for obj in (region, fun, fun.runtime, Infrastructure(region=region)):
            parsed = json.loads(obj.to_json())
            self.assertEqual(parsed, obj.to_dict())


if __name__ == '__main__':
    unittest.main()
