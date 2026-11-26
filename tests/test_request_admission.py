"""Test di aderenza dell'ammissione locale delle richieste e della coda FIFO rispetto a Serverledge Go.

Riferimenti al sistema reale (`serverledge-infrascaler`):
- `internal/scheduling/policy_default.go`: `OnArrival` e `OnCompletion` (accodamento FIFO e
  ammissione immediata dei nuovi arrivi).
- `internal/node/pool.go`: `acquireWarmContainer` e `AcquireContainer` (riuso busy concorrente,
  limite `MaxConcurrency`, riattivazione da idle).
"""

import unittest

from simulator.model.execution.container import ContainerState
from simulator.model.execution.function import Function
from simulator.model.execution.request import Request, RequestStatus
from simulator.model.execution.runtime import BuiltinRuntime
from simulator.model.topology.node import Node


class RequestAdmissionTest(unittest.TestCase):
    def test_cold_busy_reuse_limit_and_idle_reactivation(self):
        """Verifica riuso busy entro `max_concurrency`, rifiuto oltre limite e riattivazione da idle.

        Riferimento reale: `internal/node/pool.go` (`AcquireContainer`, `acquireWarmContainer`).

        - La prima richiesta avvia un cold start (`CREATING`); durante la creazione una seconda
          richiesta non può riusare il container non ancora pronto e viene rifiutata.
        - Dopo `creation_succeeded`, la seconda richiesta riusa il container busy (`warm=True`)
          senza duplicare la prenotazione CPU (`used_cpus=1`).
        - Raggiunto `max_concurrency=2`, una terza richiesta viene rifiutata (`REJECTED`).
        - Al completamento di entrambe, il container passa a `IDLE` e la richiesta successiva lo
          riattiva; tentare un doppio submit della stessa richiesta solleva `ValueError`.
        """
        node = Node('n', 256, 1, 'amd64')
        fun = Function('f', 128, 1, BuiltinRuntime('python314'), max_concurrency=2)
        first = Request(fun)
        self.assertTrue(node.submit(first))
        self.assertFalse(first.warm)
        self.assertEqual(first.container.state, ContainerState.CREATING)

        during_creation = Request(fun)
        self.assertFalse(node.submit(during_creation))

        node.creation_succeeded(first.container)
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
        """Verifica il blocco head-of-line della coda FIFO e l'ammissione immediata di nuovi arrivi.

        Riferimento reale: `internal/scheduling/policy_default.go` (`OnCompletion` estrae dalla
        FIFO solo finché la testa è eseguibile; `OnArrival` tenta l'esecuzione locale prima di accodare).

        - Mentre `active` (1 CPU) è in esecuzione, vengono accodate `oversized` (2 CPU, testa FIFO)
          e `follower` (1 CPU).
        - Al completamento di `active`, la testa `oversized` non entra nel nodo (1 CPU) e blocca
          l'estrazione anche di `follower`.
        - Un nuovo arrivo `new_arrival` (1 CPU) tenta subito l'ammissione locale e viene accettato
          sul container warm di `active`, scavalcando la FIFO bloccata esattamente come in Go.
        """
        node = Node('n', 256, 1, 'amd64', queue_capacity=2)
        fun = Function('f', 128, 1, BuiltinRuntime('python314'))
        active, follower = Request(fun), Request(fun)
        oversized = Request(Function('large', 128, 2, BuiltinRuntime('python314')))

        self.assertTrue(node.submit(active))
        node.creation_succeeded(active.container)
        self.assertTrue(node.submit(oversized))
        self.assertTrue(node.submit(follower))

        active.complete(1)
        self.assertIs(oversized.status, RequestStatus.QUEUED)
        self.assertIs(follower.status, RequestStatus.QUEUED)

        new_arrival = Request(fun)
        self.assertTrue(node.submit(new_arrival))
        self.assertIs(new_arrival.status, RequestStatus.ACCEPTED)
        self.assertIs(new_arrival.container, active.container)

    def test_fifo_lifecycle_transitions_and_rejection_when_queue_is_full(self):
        """Verifica le transizioni di stato complete (`CREATED → QUEUED → ACCEPTED → COMPLETED`) e il rifiuto a coda piena.

        Riferimento reale: `internal/scheduling/policy_default.go` e `internal/scheduling/queue.go`
        (coda a capacità finita con scarto `out_of_resources` quando piena).

        - Con `queue_capacity=1`, la seconda richiesta entra in coda (`QUEUED`), mentre la terza
          viene rifiutata con `out_of_resources`.
        - Al completamento della prima, la richiesta accodata transisce a `ACCEPTED` sullo stesso
          container e poi a `COMPLETED`, con callback osservati sempre dopo l'aggiornamento di stato.
        """
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
        node.creation_succeeded(active.container)
        self.assertTrue(node.submit(waiting))
        self.assertFalse(node.submit(rejected))
        active.complete(1)
        waiting.complete(2)

        self.assertEqual(transitions, [
            (RequestStatus.CREATED, RequestStatus.QUEUED, RequestStatus.QUEUED),
            (RequestStatus.QUEUED, RequestStatus.ACCEPTED, RequestStatus.ACCEPTED),
            (RequestStatus.ACCEPTED, RequestStatus.COMPLETED, RequestStatus.COMPLETED),
        ])


if __name__ == '__main__':
    unittest.main()
