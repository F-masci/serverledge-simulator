"""Test di aderenza del modello di funzione, container, pool e nodo rispetto a Serverledge Go.

Riferimenti al sistema reale (`serverledge-infrascaler`):
- `internal/container/runtimes.go` (`RuntimeToInfo`) e `internal/api/api.go` (`CreateOrUpdateFunction`):
  normalizzazione di `MaxConcurrency` in base al supporto alla concorrenza del runtime.
- `internal/node/node.go` (`Resources`, `FreeMemory`, `AvailableMemory`, `AvailableCPUs`):
  contabilità locale delle prenotazioni di CPU e RAM.
- `internal/node/pool.go` (`GetContainerPool`, `popIdleContainer`, `acquireWarmContainer`,
  `AcquireContainer`, `AcquireResourcesForNewContainer`, `acquireNewMemory`, `HandleCompletion`,
  `dismissContainer`, `DeleteExpiredContainer`): ciclo di vita dei container e ordine LIFO.
- `internal/container/docker.go` (`DockerFactory.Create`): traduzione dei limiti RAM e quota CPU Docker.
- `internal/test/pool_test.go`: `TestWarmCPURejectionPreservesIdleContainer`,
  `TestInvalidCPUReservationIsRejected`.
"""

import math
import unittest

from simulator.model.execution.container import ContainerState
from simulator.model.execution.function import Function
from simulator.model.execution.runtime import BuiltinRuntime, CustomRuntime
from simulator.model.topology.node import Node


def function(name='f', concurrency=2, memory=128, runtime=None):
    """Fixture: funzione da 1 CPU con compatibilità multi-architettura."""
    return Function(
        name, memory, 1, runtime or BuiltinRuntime('python314'), concurrency,
        supported_archs=('amd64', 'arm64'),
    )


class ModelsTest(unittest.TestCase):
    def setUp(self):
        # Nodo pulito per ogni test: 512 MiB di RAM e 2 CPU prenotabili.
        self.node = Node('n', 512, 2, 'amd64')

    def test_runtime_limits(self):
        """Verifica la normalizzazione di `max_concurrency` alla registrazione della funzione.

        Riferimento reale: `internal/container/runtimes.go` (`RuntimeToInfo`) e
        `internal/api/api.go` (`CreateOrUpdateFunction`).

        - I runtime concorrenti (`python314`, `python312ml`, `go125`) conservano `max_concurrency=2`.
        - I runtime non concorrenti (`nodejs17ng` e `custom`) forzano `max_concurrency=1`.
        - Un valore nullo (`concurrency=0`) viene normalizzato al minimo valido (`1`).
        """
        for name in ('python314', 'python312ml', 'go125'):
            self.assertEqual(function(runtime=BuiltinRuntime(name)).max_concurrency, 2)

        for runtime in (BuiltinRuntime('nodejs17ng'), CustomRuntime('image')):
            self.assertEqual(function(runtime=runtime).max_concurrency, 1)

        self.assertEqual(function(concurrency=0).max_concurrency, 1)

    def test_busy_reuse_and_last_completion(self):
        """Verifica il ciclo completo cold → busy condiviso → idle → scadenza TTL.

        Riferimento reale: `internal/node/pool.go` (`AcquireContainer`, `HandleCompletion`,
        `DeleteExpiredContainer`).

        - Il primo `acquire` avvia una creazione (`CREATING`) e prenota 1 CPU / 128 MiB.
        - Dopo `creation_succeeded`, un secondo `acquire` sulla stessa funzione (`max_concurrency=2`)
          riusa lo stesso container busy senza duplicare la prenotazione CPU (`used_cpus=1`).
        - Il primo `complete` lascia il container `BUSY`; solo l'ultimo `complete` (t=2) azzera la CPU
          usata e sposta la RAM nel pool warm (`warm_pool_used_mem=128`).
        - Alla scadenza esatta (`t=602`, `idle_timeout=600`) il container resta `IDLE`; strettamente
          dopo (`t=603`) il janitor lo distrugge (`DESTROYED`) liberando tutta la RAM (`free_memory=512`).
        """
        f = function()
        c = self.node.acquire(f)
        self.assertEqual(c.state, ContainerState.CREATING)
        self.node.creation_succeeded(c)
        self.assertIs(self.node.acquire(f), c)
        self.assertEqual(c.requests_count, 2)
        self.assertEqual(self.node.resources.used_cpus, 1)

        self.node.complete(c, 1)
        self.assertEqual(c.state, ContainerState.BUSY)
        self.node.complete(c, 2)
        self.assertEqual(self.node.resources.used_cpus, 0)
        self.assertEqual(self.node.resources.warm_pool_used_mem, 128)

        self.node.expire(602, container=c)
        self.assertEqual(c.state, ContainerState.IDLE)
        self.node.expire(603, container=c)
        self.assertEqual(c.state, ContainerState.DESTROYED)
        self.assertEqual(self.node.resources.free_memory, 512)

    def test_warm_cpu_rejection_preserves_idle_inventory_and_lifo(self):
        """Verifica che un rifiuto per CPU su container warm preservi inventario idle e ordine LIFO.

        Riproduce 1:1 `TestWarmCPURejectionPreservesIdleContainer` in `internal/test/pool_test.go`.

        - Due container dedicati (`a`, `b`, `concurrency=1`) completano in ordine `a` poi `b`,
          definendo lo stack idle LIFO `(a, b)` con `b` in cima.
        - Un `blocker` da 2 CPU satura tutta la CPU del nodo.
        - Un successivo `acquire(f)` viene rifiutato (`None`) senza alterare lo stack idle né i
          contatori di risorse (`used_cpus`, `busy_pool_used_mem`, `warm_pool_used_mem`).
        - Al completamento del `blocker`, il riuso warm estrae prima `b` e poi `a` (ordine LIFO).
        """
        f = function(concurrency=1)
        a, b = self.node.acquire(f), self.node.acquire(f)
        self.node.creation_succeeded(a)
        self.node.creation_succeeded(b)
        self.node.complete(a, 1)
        self.node.complete(b, 2)

        blocker = Function('blocker', 128, 2, BuiltinRuntime('python314'))
        pending = self.node.acquire(blocker)
        pool = self.node.pools[f.name]
        resources = self.node.resources
        before = (resources.used_cpus, resources.busy_pool_used_mem, resources.warm_pool_used_mem)

        self.assertIsNone(self.node.acquire(f))
        self.assertEqual(pool.idle, (a, b))
        after = (resources.used_cpus, resources.busy_pool_used_mem, resources.warm_pool_used_mem)
        self.assertEqual(after, before)

        self.node.creation_succeeded(pending)
        self.node.complete(pending, 3)
        self.assertIs(self.node.acquire(f), b)
        self.assertIs(self.node.acquire(f), a)

    def test_idle_eviction_and_docker_limits(self):
        """Verifica l'eviction della RAM warm e la traduzione dei limiti Docker.

        Riferimenti reali: `internal/node/pool.go` (`acquireNewMemory`, `dismissContainer`) e
        `internal/container/docker.go` (`DockerFactory.Create`).

        - Un container idle da 256 MiB non trattiene CPU (`used_cpus=0`).
        - L'arrivo di una funzione `g` da 512 MiB (intera RAM del nodo) forza l'eviction del
          container idle `a` (`DESTROYED`), azzerando la memoria libera residua (`free_memory=0`).
        - I limiti Docker derivati sul nuovo container sono `512 * 1048576` byte di RAM e
          `50000` microsecondi di CPU quota per 1 CPU (`cpu_period=50000` in Serverledge).
        """
        f = function(memory=256)
        a = self.node.acquire(f)
        self.node.creation_succeeded(a)
        self.node.complete(a, 1)
        self.assertEqual(self.node.resources.used_cpus, 0)

        b = self.node.acquire(function('g', memory=512))
        self.assertIsNotNone(b)
        self.assertEqual(a.state, ContainerState.DESTROYED)
        self.assertEqual(self.node.resources.free_memory, 0)
        self.assertEqual(b.memory_limit_bytes, 512 * 1048576)
        self.assertEqual(b.cpu_quota_us, 50000)

    def test_invalid_cpu_reservation_is_rejected(self):
        """Verifica che richieste di CPU non valide vengano rifiutate senza mutare le risorse.

        Riproduce `TestInvalidCPUReservationIsRejected` in `internal/test/pool_test.go`.

        - Valori di `cpu_demand` pari a `0`, `-1`, `NaN`, `+Inf`, `-Inf` vengono rifiutati in fase
          di costruzione di `Function` con `ValueError`.
        - I contatori di risorse del nodo (`used_cpus`, `free_memory`) rimangono invariati.
        """
        before = (self.node.resources.used_cpus, self.node.resources.free_memory)
        for cpu in (0, -1, math.nan, math.inf, -math.inf):
            with self.subTest(cpu=cpu), self.assertRaises(ValueError):
                Function('invalid-cpu', 128, cpu, BuiltinRuntime('python314'))
        after = (self.node.resources.used_cpus, self.node.resources.free_memory)
        self.assertEqual(after, before)

    def test_node_rejects_completion_of_another_nodes_container(self):
        """Verifica l'isolamento dei pool tra nodi distinti al completamento di un container.

        Riferimento reale: `internal/node/pool.go` (`HandleCompletion` cerca il container nel
        solo pool locale del nodo destinatario).

        - Chiamare `other.complete(c, 1)` per un container `c` allocato su `self.node` solleva
          `ValueError` e lascia inalterati stato del container e contatori di entrambi i nodi.
        """
        c = self.node.acquire(function())
        self.node.creation_succeeded(c)
        other = Node('other', 512, 2, 'amd64')
        with self.assertRaises(ValueError):
            other.complete(c, 1)
        self.assertEqual(c.state, ContainerState.BUSY)
        self.assertEqual(c.requests_count, 1)
        self.assertEqual(self.node.resources.used_cpus, 1)
        self.assertEqual(other.resources.used_cpus, 0)
        self.assertEqual(other.resources.free_memory, 512)

    def test_same_function_containers_have_distinct_identity_and_state(self):
        """Verifica che istanze distinte della stessa funzione abbiano identità e stato indipendenti.

        Riferimento reale: `internal/node/pool.go` (`GetContainerPool`, lista busy/idle per funzione).

        - Con `max_concurrency=1`, due acquisizioni concorrenti creano due container distinti (`a`, `b`)
          nel medesimo pool con ID diversi.
        - Il completamento di `a` sposta solo `a` nel pool `idle` (RAM warm), mentre `b` rimane
          `BUSY` trattenendo la propria CPU e RAM busy.
        """
        f = function(concurrency=1)
        a = self.node.acquire(f)
        self.node.creation_succeeded(a)
        b = self.node.acquire(f)
        self.node.creation_succeeded(b)

        self.assertIs(a.function, f)
        self.assertIs(b.function, f)
        self.assertIsNot(a, b)
        self.assertNotEqual(a.id, b.id)
        pool = self.node.pools[f.name]
        self.assertEqual(pool.busy, (a, b))

        self.node.complete(a, 1)
        self.assertEqual(pool.idle, (a,))
        self.assertEqual(pool.busy, (b,))
        self.assertEqual(a.state, ContainerState.IDLE)
        self.assertEqual(b.state, ContainerState.BUSY)
        self.assertEqual(self.node.resources.used_cpus, 1)
        self.assertEqual(self.node.resources.busy_pool_used_mem, f.memory_mb)
        self.assertEqual(self.node.resources.warm_pool_used_mem, f.memory_mb)


if __name__ == '__main__':
    unittest.main()
