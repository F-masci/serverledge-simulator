"""Ammissione locale delle richieste.

Riferimenti Infrascaler: internal/scheduling/policy_default.go, OnArrival;
internal/node/pool.go, acquireWarmContainer e AcquireContainer.
L'esito della creazione è esplicito. I test coprono il limite di concorrenza
e la FIFO locale configurabile.
"""

import unittest

from sim.model.node import Node
from sim.model.function import Function, BuiltinRuntime
from sim.model.request import Request, RequestStatus
from sim.model.container import ContainerState


class RequestAdmissionTest(unittest.TestCase):
    def test_cold_busy_reuse_limit_and_idle_reactivation(self):
        """Il submit usa lo stesso ciclo di acquisizione e non duplica le risorse sul busy."""
        node = Node('n', 256, 1, 'amd64')
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
        first.complete(1)
        second.complete(2)
        third = Request(fun)
        self.assertTrue(node.submit(third))
        self.assertIs(third.container, first.container)
        with self.assertRaises(ValueError):
            node.submit(third)
        self.assertEqual(third.container.requests_count, 1)

    def test_fifo_stops_at_blocked_head_and_new_arrivals_try_immediately(self):
        """Segue OnCompletion e OnArrival della policy locale default in Go."""
        node = Node('n', 256, 1, 'amd64', queue_capacity=2)
        fun = Function('f', 128, 1, BuiltinRuntime('python314'))
        active, follower = Request(fun), Request(fun)
        oversized = Request(Function('large', 128, 2, BuiltinRuntime('python314')))
        self.assertTrue(node.submit(active))
        node.creation_succeeded(active.container, 0)
        self.assertTrue(node.submit(oversized))
        self.assertTrue(node.submit(follower))

        active.complete(1)
        self.assertIs(oversized.status, RequestStatus.QUEUED)
        self.assertIs(follower.status, RequestStatus.QUEUED)
        new_arrival = Request(fun)
        self.assertTrue(node.submit(new_arrival))
        self.assertIs(new_arrival.status, RequestStatus.ACCEPTED)
        self.assertIs(new_arrival.container, active.container)

    def test_callbacks_observe_request_after_each_transition(self):
        """La FIFO ammette la richiesta e i callback vedono lo stato aggiornato."""
        node = Node('n', 128, 1, 'amd64', queue_capacity=1)
        fun = Function('f', 128, 1, BuiltinRuntime('python314'))
        active, waiting, rejected = (Request(fun) for _ in range(3))
        transitions = []
        waiting.on_queued = lambda request: self.assertIs(request.node, node)
        waiting.on_accepted = lambda request: self.assertIs(request.container, active.container)
        waiting.on_completed = lambda request: self.assertIs(request.container.state, ContainerState.IDLE)
        waiting.on_state_changed = lambda request, previous, current: transitions.append(
            (previous, current, request.status)
        )
        rejected.on_rejected = lambda request: self.assertEqual(request.rejection_reason, 'out_of_resources')

        self.assertTrue(node.submit(active))
        node.creation_succeeded(active.container, 0)
        self.assertTrue(node.submit(waiting))
        self.assertFalse(node.submit(rejected))
        active.complete(1)
        waiting.complete(2)

        self.assertEqual(transitions, [
            (RequestStatus.CREATED, RequestStatus.QUEUED, RequestStatus.QUEUED),
            (RequestStatus.QUEUED, RequestStatus.ACCEPTED, RequestStatus.ACCEPTED),
            (RequestStatus.ACCEPTED, RequestStatus.COMPLETED, RequestStatus.COMPLETED),
        ])
