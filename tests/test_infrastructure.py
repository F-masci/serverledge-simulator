"""Test degli invarianti dell'infrastruttura multi-ring e isolamento delle risorse tra nodi.

Riferimenti al sistema reale (`serverledge-infrascaler`):
- `internal/lb/generalLoadBalancer.go`: un hash ring per ciascun `machine_tag` (più ring possono
  condividere la stessa architettura ma con tag distinti).
- `internal/node/node.go`: prenotazioni CPU/RAM isolate per ciascun nodo worker.
"""

import unittest

from simulator.bindings import NodeBindings
from simulator.model.execution.function import Function
from simulator.model.execution.runtime import BuiltinRuntime
from simulator.model.topology.infrastructure import Infrastructure
from simulator.model.topology.region import Region


class InfrastructureTest(unittest.TestCase):
    def test_ring_capacities_are_shared_but_reservations_are_independent(self):
        """Verifica che ring omogenei condividano la capacità ma isolino le prenotazioni per nodo.

        Riferimento reale: `internal/lb/generalLoadBalancer.go` (unicità di `machine_tag` tra ring)
        e `internal/node/node.go` (`LocalResources` indipendenti per nodo).

        - Due ring sulla stessa architettura (`amd64`) possono coesistere se hanno `machine_tag`
          distinti (`amd64-small`, `amd64-large`); un duplicato di `machine_tag` solleva `ValueError`.
        - Tutti i nodi di uno stesso ring ereditano le medesime capacità totali, ma l'acquisizione
          di risorse su `a` non altera `used_cpus` di `b`.
        """
        infra = Infrastructure()
        small = infra.create_ring('small', arch='amd64', memory=512, cpus=2, machine_tag='amd64-small')
        large = infra.create_ring('large', arch='amd64', memory=1024, cpus=4, machine_tag='amd64-large')
        with self.assertRaisesRegex(ValueError, 'machine_tag'):
            infra.create_ring('dup', arch='amd64', memory=512, cpus=2, machine_tag='amd64-small')

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
        """Verifica che creazioni invalide o conflittuali lascino intatti topologia e inventario.

        - La creazione di un nodo duplicato, su un ring inesistente o di un ring con memoria non
          valida (`0`, `-1`, `1.5`, `True`) solleva `ValueError`.
        - Dopo i tentativi falliti, l'insieme dei ring, i nodi registrati e le capacità rimangono
          invariati.
        """
        infra = Infrastructure()
        ring = infra.create_ring('small', arch='amd64', memory=512, cpus=2)
        node = infra.create_node('small', 'n1')
        with self.assertRaises(ValueError):
            infra.create_node('small', 'n1')
        with self.assertRaises(ValueError):
            infra.create_node('missing', 'n2')
        for memory in (0, -1, 1.5, True):
            with self.subTest(memory=memory), self.assertRaises(ValueError):
                infra.create_ring('bad', arch='amd64', memory=memory, cpus=2)
        self.assertEqual(tuple(infra.rings), ('small',))
        self.assertEqual(tuple(ring), (node,))
        self.assertEqual(node.resources.total_cpus, 2)

    def test_node_bindings_are_propagated_to_dynamically_created_rings_and_nodes(self):
        """Verifica che ring e nodi creati dinamicamente a runtime (scale-out) ereditino i `NodeBindings`.

        - I nodi aggiunti sia all'avvio (`r1/n1`) sia dinamicamente in un secondo ring (`r2/n2`)
          ricevono automaticamente i callback `on_node_registered` e `on_container_state_changed`.
        """
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
        infra.create_ring('r2', arch='arm64', memory=256, cpus=1)
        n2 = infra.create_node('r2', 'n2')

        self.assertEqual(registered, [('n1', 'edge', 'r1'), ('n2', 'edge', 'r2')])
        fun = Function('f', 128, 1, BuiltinRuntime('python314'))
        n1.acquire(fun)
        n2.acquire(fun)
        self.assertEqual([(e[0], e[1], e[2]) for e in state_events], [('edge', 'r1', 'n1'), ('edge', 'r2', 'n2')])


if __name__ == '__main__':
    unittest.main()
