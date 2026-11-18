"""Ammissione locale delle richieste.

Riferimenti Infrascaler: internal/scheduling/policy_default.go, OnArrival;
internal/node/pool.go, acquireWarmContainer e AcquireContainer.
La coda è disabilitata: senza risorse si rifiuta. La creazione viene riservata,
ma il suo esito resta esplicito, senza simulare il servizio in questo passo.
"""

import unittest

from sim.model.node import Node
from sim.model.function import Function, BuiltinRuntime
from sim.model.request import Request, RequestStatus
from sim.model.container import ContainerState


class RequestAdmissionTest(unittest.TestCase):
    def test_cold_busy_reuse_limit_and_idle_reactivation(self):
        """Il submit usa lo stesso ciclo di acquisizione e non duplica le risorse sul busy."""
        node = Node('n', 256, 1, 'edge', 'amd64')
        fun = Function('f', 128, 1, BuiltinRuntime('python314'), max_concurrency=2)
        first = Request(fun)
        self.assertTrue(node.submit(first))
        self.assertFalse(first.warm)
        self.assertEqual(first.container.state, ContainerState.CREATING)
        during_creation = Request(fun)
        self.assertFalse(node.submit(during_creation))
        node.creation_succeeded(first.container, 0)
        second = Request(fun)
        self.assertTrue(node.submit(second))
        self.assertTrue(second.warm)
        self.assertIs(second.container, first.container)
        self.assertEqual(node.resources.used_cpus, 1)
        rejected = Request(fun)
        self.assertFalse(node.submit(rejected))
        self.assertEqual(rejected.status, RequestStatus.REJECTED)
        node.complete(first.container, 1)
        node.complete(second.container, 2)
        third = Request(fun)
        self.assertTrue(node.submit(third))
        self.assertIs(third.container, first.container)
        with self.assertRaises(ValueError):
            node.submit(third)
        self.assertEqual(third.container.requests_count, 1)
