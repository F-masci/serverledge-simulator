"""Percorso di inoltro iniziale e compatibilità dell'hashing.

Riferimento Cortese: internal/lb/hashRing.go, hash, HashRing.Add e HashRing.Get;
internal/lb/architectureAwareLb.go, NewArchitectureAwareBalancer (128 repliche).
FNV-1a, chiavi nome#indice, successore inclusivo e wrap-around seguono quel codice.
La selezione del primo ring compatibile e l’ammissione locale senza coda sono
semplificazioni esplicite: non sono riprodotti MemoryChecker, fallback per
risorse, MAB o una coda di ammissione finita della piattaforma reale.
"""

import unittest
from unittest.mock import patch

from sim.model.infrastructure import Infrastructure
from sim.model.request import Request, RequestStatus
from sim.model.function import Function, BuiltinRuntime
from sim.model.ring import ArchitectureRing


def function(name='hello', archs=('amd64', 'arm64')):
    return Function(name, 128, 1, BuiltinRuntime('python314'), supported_archs=archs)


class RoutingTest(unittest.TestCase):
    def test_fnv1a_matches_known_32_bit_vectors(self):
        """Valori canonici impediscono di usare accidentalmente FNV-1 o hash Python."""
        self.assertEqual(ArchitectureRing._hash(''), 0x811C9DC5)
        self.assertEqual(ArchitectureRing._hash('a'), 0xE40C292C)
        self.assertEqual(ArchitectureRing._hash('foobar'), 0xBF9CF968)

    def test_infrastructure_delegates_and_node_admits_request(self):
        """La stessa istanza percorre LB/ring/nodo e ottiene la prenotazione locale."""
        infra = Infrastructure()
        ring = infra.create_ring('small', arch='amd64', memory=512, cpus=2)
        node = infra.create_node('small', 'n1', 'edge')
        requests = [Request(function()), Request(function())]
        with patch.object(infra.load_balancer, 'submit', wraps=infra.load_balancer.submit) as lb:
            with patch.object(ring, 'submit', wraps=ring.submit) as submit:
                for request in requests:
                    self.assertIs(infra.submit(request), node)
        self.assertEqual(lb.call_count, 2)
        self.assertEqual(submit.call_count, 2)
        for request in requests:
            self.assertIs(request.node, node)
            self.assertIs(request.function, request.container.function)
            self.assertEqual(request.status, RequestStatus.ACCEPTED)
        self.assertEqual(node.resources.used_cpus, 2)
        self.assertEqual(node.resources.free_memory, 256)

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

    def test_architecture_is_checked_once_when_selecting_the_ring(self):
        """Il LB filtra il ring; hashing e nodo si affidano all'omogeneità garantita."""
        infra = Infrastructure()
        infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        node = infra.create_node('r', 'n', 'edge')
        fun = function()
        with patch.object(Function, 'supports_arch', autospec=True, return_value=True) as supports:
            self.assertIs(infra.submit(Request(fun)), node)
        supports.assert_called_once_with(fun, 'amd64')

    def test_hash_lookup_uses_inclusive_successor_and_wraparound(self):
        """I punti sono hash(nome#indice); il confine incluso e il ritorno all'inizio sono esatti."""
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2, replicas=2)
        nodes = [infra.create_node('r', name, 'edge') for name in ('n1', 'n2')]
        points = sorted((ArchitectureRing._hash(f'{node.id.key}#{i}'), node) for node in nodes for i in range(2))
        for query, expected in ((points[1][0], points[1][1]),
                                (points[1][0] + 1, points[2][1]),
                                (points[-1][0] + 1, points[0][1])):
            with patch.object(ArchitectureRing, '_hash', return_value=query):
                self.assertIs(ring.select_node(function()), expected)

    def test_same_function_routes_consistently_without_using_request_identity(self):
        """Il routing dipende dal nome della funzione, non dall'ID dell'invocazione."""
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        for name in ('one', 'two', 'three'):
            infra.create_node('r', name, 'edge')
        self.assertEqual(ring.replicas, 128)
        target = ring.select_node(function())
        for _ in range(2):
            self.assertIs(infra.submit(Request(function())), target)
        rejected = Request(function())
        self.assertIsNone(infra.submit(rejected))
        self.assertEqual(rejected.rejection_reason, 'out_of_resources')


if __name__ == '__main__':
    unittest.main()
