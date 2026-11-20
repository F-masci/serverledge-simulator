"""Test dei modelli di funzione, container, pool e nodo.

Riferimenti al codice Infrascaler (percorsi relativi al repository):
- internal/container/runtimes.go, RuntimeToInfo: supporto alla concorrenza.
- internal/api/api.go, CreateOrUpdateFunction: normalizzazione di MaxConcurrency.
- internal/node/node.go, Resources: contabilità locale; FreeMemory,
  AvailableMemory e AvailableCPUs: risorse libere e recuperabili.
- internal/node/pool.go, GetContainerPool: pool locale per funzione;
  popIdleContainer: ordine LIFO; acquireWarmContainer e AcquireContainer:
  riuso e acquisizione; AcquireResourcesForNewContainer e acquireNewMemory:
  prenotazioni CPU/RAM; HandleCompletion: rilascio all'ultima invocazione;
  NewContainerWithAcquiredResources: pubblicazione e rollback della creazione;
  PrewarmInstances: creazione anticipata; dismissContainer: recupero RAM idle;
  DeleteExpiredContainer: eliminazione dopo la scadenza.
- internal/container/docker.go, DockerFactory.Create: limiti RAM e quota CPU.
"""

import unittest

from sim.model.function import Function, BuiltinRuntime, CustomRuntime
from sim.model.node import Node
from sim.model.container import ContainerState


def function(name='f', concurrency=2, memory=128, runtime=None):
    """Fixture: funzione da una CPU, con compatibilità già risolta."""
    return Function(name, memory, 1, runtime or BuiltinRuntime('python314'), concurrency, supported_archs=('amd64', 'arm64'))


class ModelsTest(unittest.TestCase):
    def setUp(self):
        # Nodo nuovo per ogni test: 512 unità di RAM e 2 CPU prenotabili.
        self.node = Node('n', 512, 2, 'amd64')

    def test_runtime_limits(self):
        """Verifica il limite effettivo alla registrazione della funzione.

        I runtime concorrenti mantengono la richiesta di due invocazioni;
        nodejs17ng e custom la riducono a una. Un valore zero diventa uno.
        """
        for name in ('python314', 'python312ml', 'go125'):
            self.assertEqual(function(runtime=BuiltinRuntime(name)).max_concurrency, 2)

        for runtime in (BuiltinRuntime('nodejs17ng'), CustomRuntime('image')):
            self.assertEqual(function(runtime=runtime).max_concurrency, 1)

        self.assertEqual(function(concurrency=0).max_concurrency, 1)

    def test_busy_reuse_and_last_completion(self):
        """Verifica il ciclo cold → busy condiviso → idle → scadenza.

        Il secondo acquire deve restituire la stessa istanza senza una seconda
        prenotazione CPU. Solo l'ultimo completamento libera CPU e rende warm
        la RAM. Alla scadenza esatta il container resta; dopo viene rimosso.
        CREATING è uno stato del simulatore: rende visibile la prenotazione
        che precede la pubblicazione del container nel pool reale.
        """
        f = function()
        c = self.node.acquire(f)
        self.assertEqual(c.state, ContainerState.CREATING)
        self.node.creation_succeeded(c, 0)
        self.assertIs(self.node.acquire(f), c)
        self.assertEqual(c.requests_count, 2)
        self.assertEqual(self.node.resources.used_cpus, 1)
        # La prima conclusione lascia una richiesta attiva e il container busy.
        self.node.complete(c, 1)
        self.assertEqual(c.state, ContainerState.BUSY)
        self.node.complete(c, 2, timeout=10)
        self.assertEqual(self.node.resources.used_cpus, 0)
        self.assertEqual(self.node.resources.warm_pool_used_mem, 128)
        # Ultimo completamento a t=2, timeout=10: confronto stretto now > 12.
        self.node.expire(12)
        self.assertEqual(c.state, ContainerState.IDLE)
        self.node.expire(13)
        self.assertEqual(c.state, ContainerState.DESTROYED)
        self.assertEqual(self.node.resources.free_memory, 512)

    def test_warm_cpu_rejection_preserves_idle_inventory_and_lifo(self):
        """Riproduce TestWarmCPURejectionPreservesIdleContainer di Infrascaler."""
        f = function(concurrency=1)
        a, b = self.node.acquire(f), self.node.acquire(f)
        self.node.creation_succeeded(a, 0)
        self.node.creation_succeeded(b, 0)
        self.node.complete(a, 1)
        self.node.complete(b, 2)

        blocker = Function('blocker', 128, 2, BuiltinRuntime('python314'))
        pending = self.node.acquire(blocker)
        pool = self.node.pools[f.name]
        resources = self.node.resources
        before = (resources.used_cpus, resources.busy_pool_used_mem, resources.warm_pool_used_mem)

        self.assertIsNone(self.node.acquire(f, only_warm=True))
        self.assertEqual(pool.idle, (a, b))
        self.assertEqual((resources.used_cpus, resources.busy_pool_used_mem, resources.warm_pool_used_mem), before)

        self.node.creation_failed(pending)
        self.assertIs(self.node.acquire(f, only_warm=True), b)
        self.assertIs(self.node.acquire(f, only_warm=True), a)

    def test_failed_creation_rolls_back_both_buckets(self):
        """Verifica il rollback di una creazione fallita normale o prewarm.

        Una creazione normale prenota RAM busy e CPU; una prewarm prenota RAM
        warm. In entrambi i casi il fallimento deve restituire tutta la RAM e
        lasciare zero CPU prenotate. DESTROYED indica un oggetto simulato non
        più utilizzabile: non attesta la pulizia di oggetti Docker parziali.
        """
        # Stessa verifica sui due diversi bucket di prenotazione RAM.
        for prewarm in (False, True):
            with self.subTest(prewarm=prewarm):
                c = self.node.prewarm(function()) if prewarm else self.node.acquire(function())
                self.node.creation_failed(c)
                self.assertEqual(c.state, ContainerState.DESTROYED)
                self.assertEqual(self.node.resources.free_memory, 512)
                self.assertEqual(self.node.resources.used_cpus, 0)

    def test_prewarm_eviction_and_limits(self):
        """Verifica prewarming, recupero RAM warm e traduzione dei limiti Docker.

        Un prewarm da 256 non trattiene CPU. Una nuova funzione da 512 richiede
        tutta la memoria del nodo: il container idle deve essere eliminato.
        La nuova prenotazione occupa tutta la RAM. I limiti Docker derivati sono
        512 x 1048576 byte e 50000 microsecondi di quota per una CPU.
        Si assume distruzione riuscita; non vengono
        verificati errori Docker o ordine di eviction fra più candidati.
        """
        f = function(memory=256)
        a = self.node.prewarm(f)
        self.node.creation_succeeded(a, 0)
        self.assertEqual(self.node.resources.used_cpus, 0)
        b = self.node.acquire(function('g', memory=512))
        self.assertIsNotNone(b)
        self.assertEqual(a.state, ContainerState.DESTROYED)
        self.assertEqual(self.node.resources.free_memory, 0)
        self.assertEqual(b.memory_limit_bytes, 512 * 1048576)
        self.assertEqual(b.cpu_quota_us, 50000)

    def test_pool_supports_length_membership_and_iteration(self):
        """Il pool espone numero e istanze dei container tramite protocolli Python.

        Dopo la pubblicazione, len conta il container, in lo riconosce e
        l'iterazione restituisce la stessa istanza registrata.
        """
        c = self.node.acquire(function())
        self.node.creation_succeeded(c, 0)
        pool = self.node.pools['f']
        self.assertEqual(len(pool), 1)
        self.assertIn(c, pool)
        self.assertEqual(tuple(pool), (c,))

    def test_public_state_rejects_direct_assignment(self):
        """Contatore richieste e mappa dei pool sono pubblicamente di sola lettura.

        Le assegnazioni dirette devono fallire: le modifiche passano dalle
        operazioni del nodo per mantenere coerenti stato e prenotazioni.
        Queste eccezioni sono un contratto Python del modello.
        """
        c = self.node.acquire(function())
        self.node.creation_succeeded(c, 0)
        pool = self.node.pools['f']
        with self.assertRaises(AttributeError):
            c.requests_count = 10
        with self.assertRaises(TypeError):
            self.node.pools['other'] = pool
        self.assertEqual(c.requests_count, 1)
        self.assertEqual(tuple(self.node.pools), ('f',))

    def test_node_rejects_completion_of_another_nodes_container(self):
        """Un nodo non può completare una richiesta ospitata da un altro nodo.

        Il rifiuto non deve cambiare il container né le risorse dei due nodi.
        Il modello solleva ValueError; il codice Go registra il mancato
        ritrovamento nel pool senza esporre questa eccezione Python.
        """
        c = self.node.acquire(function())
        self.node.creation_succeeded(c, 0)
        other = Node('other', 512, 2, 'amd64')
        with self.assertRaises(ValueError):
            other.complete(c, 1)
        self.assertEqual(c.state, ContainerState.BUSY)
        self.assertEqual(c.requests_count, 1)
        self.assertEqual(self.node.resources.used_cpus, 1)
        self.assertEqual(other.resources.used_cpus, 0)
        self.assertEqual(other.resources.free_memory, 512)

    def test_same_function_containers_have_distinct_identity_and_state(self):
        """Due istanze della stessa funzione restano distinte nel pool.

        Con concorrenza uno, una seconda richiesta mentre la prima è attiva
        crea un altro container. Funzione condivisa non significa identità
        condivisa: oggetti e ID devono differire, senza collassare in un set.
        Completando il primo, soltanto quello diventa idle; il secondo resta
        busy con la propria richiesta e la propria prenotazione CPU/RAM.
        """
        f = function(concurrency=1)
        a = self.node.acquire(f)
        self.node.creation_succeeded(a, 0)
        b = self.node.acquire(f)
        self.node.creation_succeeded(b, 0)

        self.assertIs(a.function, f)
        self.assertIs(b.function, f)
        self.assertIsNot(a, b)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a.id, b.id)
        self.assertEqual(len({a, b}), 2)
        pool = self.node.pools[f.name]
        self.assertEqual(len(pool), 2)
        self.assertIn(a, pool)
        self.assertIn(b, pool)
        self.assertEqual(pool.busy, (a, b))

        # Il completamento individua l'istanza, non soltanto la funzione.
        self.node.complete(a, 1)
        self.assertEqual(pool.idle, (a,))
        self.assertEqual(pool.busy, (b,))
        self.assertEqual(a.requests_count, 0)
        self.assertEqual(b.requests_count, 1)
        self.assertEqual(a.state, ContainerState.IDLE)
        self.assertEqual(b.state, ContainerState.BUSY)
        self.assertEqual(self.node.resources.used_cpus, 1)
        self.assertEqual(self.node.resources.busy_pool_used_mem, f.memory_mb)
        self.assertEqual(self.node.resources.warm_pool_used_mem, f.memory_mb)


if __name__ == '__main__':
    unittest.main()
