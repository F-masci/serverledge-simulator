"""Test di aderenza delle policy di scheduling e offloading rispetto a Serverledge Go.

Riferimenti al sistema reale (`serverledge-infrascaler`):
- `internal/scheduling/policy_default.go`: accodamento FIFO locale senza offloading.
- `internal/scheduling/edgeOnlyPolicy.go`: offloading verso vicini edge e fallback locale senza FIFO.
- `internal/scheduling/edgeCloudPolicy.go`: esecuzione locale prioritaria con offload verso regione cloud.
- `internal/scheduling/cloudonly_policy.go`: inoltro esclusivo verso regione cloud.
- `internal/scheduling/offloading.go`: `pickEdgeNodeForOffloading` (scelta su memoria positiva) e
  `offloadingCache` (TTL 60s).
- `internal/test/scheduling_test.go`: `TestMismatchingArchNoOffload`, `TestEdgeOnlyLocalRejectionsComplete`.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario
from simulator.model.execution.function import Function
from simulator.model.execution.request import Request, RequestStatus
from simulator.model.execution.runtime import BuiltinRuntime
from simulator.model.scheduling.offloading import NodeScheduling, OffloadingScheduler, SchedulingPolicy
from simulator.model.topology.infrastructure import Infrastructure
from simulator.model.topology.region import Region


class OffloadingTest(unittest.TestCase):
    def setUp(self):
        self.infrastructure = Infrastructure()
        self.function = Function('f', 128, 1, BuiltinRuntime('python314'), supported_archs=('amd64',))

    def node(self, name, *, arch='amd64', tag=None, memory=128, cpus=1, queue_capacity=0):
        ring_tag = tag or f'{arch}/{name}'
        self.infrastructure.create_ring(name, arch=arch, memory=memory, cpus=cpus, machine_tag=ring_tag)
        return self.infrastructure.create_node(name, name, ring_tag, queue_capacity=queue_capacity)

    def test_default_policy_queues_locally_even_if_offloading_is_allowed(self):
        """Verifica che la policy `default` accodi localmente senza offload anche con `can_offload=True`.

        Riferimento reale: `internal/scheduling/policy_default.go` (`OnArrival` e `OnCompletion`).

        - Un nodo saturo con `queue_capacity=1` accoda la seconda richiesta (`QUEUED`) senza offload.
        - Al completamento della prima richiesta, la richiesta accodata viene ammessa (`ACCEPTED`)
          riutilizzando il medesimo container warm.
        """
        origin = self.node('origin', queue_capacity=1)
        active, waiting = (Request(self.function, can_offload=True) for _ in range(2))
        self.assertIs(self.infrastructure.submit(active, node=origin), origin)
        origin.creation_succeeded(active.container)
        self.assertIs(self.infrastructure.submit(waiting, node=origin), origin)
        self.assertIs(waiting.status, RequestStatus.QUEUED)
        self.assertIsNone(waiting.to_dict()['offloaded_from_node_id'])

        active.complete(1)
        self.assertIs(waiting.status, RequestStatus.ACCEPTED)
        self.assertIs(waiting.container, active.container)

    def test_mismatching_arch_and_oversized_cpu_reject_locally(self):
        """Verifica il rifiuto locale per architettura incompatibile o CPU eccessiva.

        Riproduce `TestMismatchingArchNoOffload` e `TestEdgeOnlyLocalRejectionsComplete` in
        `internal/test/scheduling_test.go`.

        - Policy `default` con `can_offload=False`: una funzione `arm64` su nodo `amd64` viene
          rifiutata (`REJECTED`) senza allocare risorse.
        - Policy `edgeonly` con `fallback_local=True` senza vicini idonei: sia un mismatch di
          architettura (`arm64` su `amd64`) sia una domanda CPU superiore alla capacità del nodo
          terminano con rifiuto `out_of_resources`/`incompatible_architecture` senza bloccarsi.
        """
        origin = self.node('origin', arch='amd64', memory=256, cpus=1)
        arm_fn = Function('arm-only', 128, 1, BuiltinRuntime('python314'), supported_archs=('arm64',))
        huge_fn = Function('huge-cpu', 128, 999, BuiltinRuntime('python314'), supported_archs=('amd64',))

        # TestMismatchingArchNoOffload (default policy, can_offload=False).
        no_offload = Request(arm_fn, can_offload=False)
        self.assertIsNone(self.infrastructure.submit(no_offload, node=origin))
        self.assertIs(no_offload.status, RequestStatus.REJECTED)

        # TestEdgeOnlyLocalRejectionsComplete (edgeonly policy).
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True,
        ))
        for label, fn in (('architecture', arm_fn), ('capacity', huge_fn)):
            with self.subTest(case=label):
                req = Request(fn, can_offload=True)
                self.assertIsNone(self.infrastructure.submit(req, node=origin))
                self.assertIs(req.status, RequestStatus.REJECTED)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_edgeonly_falls_back_locally_when_no_compatible_neighbor_exists(self):
        """Verifica il fallback locale di `edgeonly` in assenza di vicini o con vicini incompatibili.

        Riferimento reale: `internal/scheduling/edgeOnlyPolicy.go` (`OnArrival` con `fallbackLocal=true`).

        - Con un vicino di architettura incompatibile (`arm64`) o con zero vicini nella regione,
          `edgeonly` esegue immediatamente sul nodo di origine senza marcare offload.
        """
        # Caso 1: zero vicini scoperti.
        solo = self.node('solo')
        self.infrastructure.configure_scheduler(solo, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True,
        ))
        req_solo = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(req_solo, node=solo), solo)
        self.assertIs(req_solo.status, RequestStatus.ACCEPTED)
        self.assertIsNone(req_solo.to_dict()['offloaded_from_node_id'])

        # Caso 2: solo un vicino incompatibile per architettura.
        origin = self.node('origin')
        peer = self.node('arm', arch='arm64')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(peer,),
        ))
        req_fallback = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(req_fallback, node=origin), origin)
        self.assertIsNone(req_fallback.to_dict()['offloaded_from_node_id'])

    def test_edgeonly_local_fallback_does_not_use_the_fifo(self):
        """Verifica che il fallback locale di `edgeonly` non accodi nella FIFO locale.

        Riferimento reale: `internal/scheduling/edgeOnlyPolicy.go` (`handleLocalExecution` con
        accodamento disabilitato nel fallback edge-only).

        - Anche con `queue_capacity=1`, se il nodo di origine è saturo e non esistono vicini
          compatibili, la richiesta viene rifiutata immediatamente (`REJECTED`) anziché accodata.
        """
        origin = self.node('origin', queue_capacity=1)
        peer = self.node('arm', arch='arm64')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(peer,),
        ))

        active = Request(self.function)
        fallback = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(active, node=origin), origin)
        self.assertIsNone(self.infrastructure.submit(fallback, node=origin))
        self.assertIs(fallback.status, RequestStatus.REJECTED)
        self.assertIs(fallback.node, origin)

    def test_edgeonly_uses_offloading_cache_and_reevaluates_after_expiration(self):
        """Verifica il riuso della cache di offloading per funzione (TTL 60s) e la rivalutazione a scadenza.

        Riferimento reale: `internal/scheduling/offloading.go` (`offloadingCache`, validità 60s).

        - A t=20 `origin` sceglie `large` (più memoria libera di `small`) e imposta `can_offload=False`
          per impedire multi-hop.
        - A t=21 (entro 60s), anche dopo che `blocker` ha saturato `large`, la cache continua a
          inoltrare a `large` (che rifiuta).
        - A t=81 (oltre 60s), la cache è scaduta: `origin` rivaluta i vicini e inoltra a `small`.
        """
        origin = self.node('origin', queue_capacity=1)
        small = self.node('small', memory=256)
        large = self.node('large', memory=512, cpus=2)
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(small, large),
        ))

        first = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(first, node=origin, now=20), large)
        self.assertEqual(first.to_dict()['offloaded_from_node_id'], origin.id.key)
        self.assertFalse(first.can_offload)
        blocker = Function('blocker', 384, 1, BuiltinRuntime('python314'))
        self.assertIs(self.infrastructure.submit(Request(blocker), node=large, now=21), large)

        cached = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(cached, node=origin, now=21))
        self.assertIs(cached.node, large)

        after_expiry = Request(self.function, can_offload=True)
        self.assertIs(self.infrastructure.submit(after_expiry, node=origin, now=81), small)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_edgeonly_forwards_to_positive_memory_even_if_insufficient(self):
        """Verifica che `edgeonly` inoltri al vicino con memoria positiva anche se insufficiente per la funzione.

        Riferimento reale: `internal/scheduling/offloading.go` (`pickEdgeNodeForOffloading` confronta
        solo `AvailableMemory > 0` senza verificare `AvailableMemory >= fun.MemoryMB`).

        - `peer` ha 64 MiB liberi (`> 0` ma `< 128 MiB`): `origin` inoltra comunque la richiesta a
          `peer`, che la rifiuta localmente lasciando intatte le risorse di `origin`.
        """
        origin = self.node('origin')
        peer = self.node('peer', memory=64)
        self.infrastructure.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=True, nearest_neighbors=(peer,),
        ))

        request = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(request, node=origin, now=20))
        self.assertIs(request.node, peer)
        self.assertEqual(request.to_dict()['offloaded_from_node_id'], origin.id.key)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_edgecloud_runs_locally_first_then_offloads_to_configured_region(self):
        """Verifica che `edgecloud` esegua prima in locale e inoltri al cloud solo a saturazione.

        Riferimento reale: `internal/scheduling/edgeCloudPolicy.go` (`OnArrival`).

        - La prima richiesta viene eseguita localmente su `edge`.
        - A nodo saturo, una richiesta con `can_offload=False` viene rifiutata, mentre una con
          `can_offload=True` viene selezionata per l'offload verso la regione `cloud`.
        """
        edge = self.node('edge', queue_capacity=1)
        cloud = Region('cloud')
        self.infrastructure.configure_scheduler(edge, NodeScheduling(
            SchedulingPolicy.EDGE_CLOUD, remote_region=cloud,
        ))
        local = Request(self.function, can_offload=True)
        forbidden = Request(self.function)
        self.assertIs(self.infrastructure.submit(local, node=edge), edge)
        self.assertIsNone(self.infrastructure.submit(forbidden, node=edge))
        self.assertIs(forbidden.status, RequestStatus.REJECTED)

        scheduler = OffloadingScheduler.create(
            lambda: (edge,), NodeScheduling(SchedulingPolicy.EDGE_CLOUD, remote_region=cloud),
        )
        remote = Request(self.function, can_offload=True)
        self.assertIs(scheduler.schedule(remote, edge), cloud)
        self.assertIs(remote.status, RequestStatus.CREATED)

    def test_cloudonly_rejects_without_permission_or_remote_region(self):
        """Verifica che `cloudonly` non esegua mai localmente e rifiuti se manca il permesso o il cloud.

        Riferimento reale: `internal/scheduling/cloudonly_policy.go` (`OnArrival`).

        - Senza `remote_region` configurata, sia `can_offload=False` sia `can_offload=True`
          vengono rifiutati (`REJECTED`) senza occupare risorse locali.
        """
        origin = self.node('origin')
        self.infrastructure.configure_scheduler(origin, NodeScheduling(SchedulingPolicy.CLOUD_ONLY))
        for can_offload in (False, True):
            with self.subTest(can_offload=can_offload):
                request = Request(self.function, can_offload=can_offload)
                self.assertIsNone(self.infrastructure.submit(request, node=origin))
                self.assertIs(request.status, RequestStatus.REJECTED)
        self.assertEqual(origin.resources.used_cpus, 0)

    def test_local_rejection_after_proxy_selection_does_not_retry_another_node(self):
        """Verifica che il reverse proxy non ritenti su un altro nodo dopo il rifiuto dello scheduler.

        Riferimento reale: `internal/lb/lb.go` (una volta scelto il target dal proxy, l'eventuale
        HTTP 429/500 restituito dal worker torna direttamente al client senza retry).

        - Il nodo primario del ring usa `CLOUD_ONLY` (senza cloud) e rifiuta la richiesta.
        - Nonostante esista un secondo nodo libero nel ring, il LB non ritenta e propaga il rifiuto.
        """
        ring = self.infrastructure.create_ring('edge', arch='amd64', memory=128, cpus=1)
        for name in ('one', 'two'):
            self.infrastructure.create_node('edge', name)
        primary = ring.select_node(self.function)
        self.infrastructure.configure_scheduler(primary, NodeScheduling(SchedulingPolicy.CLOUD_ONLY))

        request = Request(self.function)
        self.assertIsNone(self.infrastructure.submit(request))
        self.assertIs(request.node, primary)
        self.assertIs(request.status, RequestStatus.REJECTED)
        self.assertTrue(all(node.resources.used_cpus == 0 for node in ring))

    def test_scenario_offloads_the_first_invocation_without_waiting_for_discovery(self):
        """Verifica end-to-end da scenario TOML l'offload immediato della prima invocazione `edgeonly`.

        Riferimento reale: `internal/scheduling/edgeOnlyPolicy.go` con topologia nota a t=0.

        - Carica uno scenario TOML con `entry_node="edge-1"`, `policy="edgeonly"` e vicino `edge-2`.
        - Al primo arrivo (t=0) la richiesta viene inoltrata e accettata direttamente su `edge-2`,
          con `offloaded_from_node_id == "edge-1"`.
        """
        scenario_data = """
[scheduler]
policy = "edgeonly"

[invocation]
entry_node = "edge-1"
can_offload = true

[[functions]]
name = "f"
runtime = "python314"
memory_mb = 128
cpu_demand = 1
arrival_rate = 1
service_rate = 1
first_arrival_time = 0
arrival_distribution = "deterministic"
service_distribution = "deterministic"

[[infrastructure.rings]]
id = "amd64"
arch = "amd64"
memory_mb = 256
cpus = 1

[[infrastructure.rings.nodes]]
name = "edge-1"
nearest_neighbors = ["edge-2"]

[[infrastructure.rings.nodes]]
name = "edge-2"
"""
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'offloading.toml'
            path.write_text(scenario_data, encoding='utf-8')
            simulation = Simulation(Scenario.load(path))
        requests = []
        infrastructure = simulation.regions['default'].infrastructure
        infrastructure.on_request_submitted = lambda _, request, node: requests.append(request)

        statistics = simulation.run(arrivals=1)
        accepted, = requests
        self.assertIs(accepted.status, RequestStatus.ACCEPTED)
        self.assertEqual(accepted.node.id.key, 'edge-2')
        self.assertEqual(accepted.to_dict()['offloaded_from_node_id'], 'edge-1')
        self.assertEqual(statistics.metrics['total']['arrivals'], 1)

    def test_edgeonly_without_local_fallback_and_dynamic_peer_discovery_and_no_cache(self):
        """Verifica `edgeonly` con `fallback_local=False`, discovery dinamica dei primi 2 peer e cache disabilitata.

        Riferimento reale: `internal/scheduling/edgeOnlyPolicy.go` (`fallbackLocal=false` rifiuta
        se nessun vicino è disponibile) e `internal/scheduling/offloading.go` (selezione sui primi
        due peer della regione quando `nearest_neighbors` non è fissato, e rivalutazione immediata
        con `offloading_cache_validity_s=0`).
        """
        # 1. fallback_local=False senza vicini compatibili -> rifiuta invece di eseguire localmente.
        isolated = self.node('isolated')
        self.infrastructure.configure_scheduler(isolated, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=False, nearest_neighbors=(),
        ))
        req_rejected = Request(self.function, can_offload=True)
        self.assertIsNone(self.infrastructure.submit(req_rejected, node=isolated))
        self.assertIs(req_rejected.status, RequestStatus.REJECTED)
        self.assertEqual(isolated.resources.used_cpus, 0)

        # 2. Discovery dinamica (nearest_neighbors=None) con cache disabilitata (validity=0):
        #    sceglie il peer con più memoria libera tra i primi due e rivaluta subito alla richiesta successiva.
        infra2 = Infrastructure()
        for name, mem in (('origin', 128), ('p1', 256), ('p2', 512)):
            infra2.create_ring(name, arch='amd64', memory=mem, cpus=2, machine_tag=f'amd64/{name}')
            infra2.create_node(name, name, f'amd64/{name}')
        origin = infra2.resolve_node('origin')
        p1 = infra2.resolve_node('p1')
        p2 = infra2.resolve_node('p2')
        infra2.configure_scheduler(origin, NodeScheduling(
            SchedulingPolicy.EDGE_ONLY, fallback_local=False, offloading_cache_validity_s=0.0,
        ))

        r1 = Request(self.function, can_offload=True)
        self.assertIs(infra2.submit(r1, node=origin, now=1.0), p2)
        # Satura p2 con un blocker da 384 MiB (restano 0 MiB su p2, 256 MiB su p1).
        blocker = Function('blocker', 384, 1, BuiltinRuntime('python314'))
        self.assertIs(infra2.submit(Request(blocker), node=p2, now=1.5), p2)
        # Senza cache, alla richiesta successiva (t=1.6) rivaluta immediatamente e sceglie p1.
        r2 = Request(self.function, can_offload=True)
        self.assertIs(infra2.submit(r2, node=origin, now=1.6), p1)


if __name__ == '__main__':
    unittest.main()
