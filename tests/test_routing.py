"""Test di aderenza del load balancer, hash ring e cache risorse rispetto a Serverledge Go.

Riferimenti al sistema reale (`serverledge-infrascaler`):
- `internal/lb/hashRing.go`: `hash` (FNV-1a), `HashRing.Add`, `HashRing.Get` (repliche virtuali,
  wraparound e fallback intra-ring).
- `internal/lb/generalLoadBalancer.go`: filtro per `arch` e `machine_tag`, ordine di scoperta dei
  ring, fallback inter-ring compatibile, strategie `Random`, `RoundRobin`, `ArchitectureUnaware`.
- `internal/lb/lb.go`: cache locale `NodeMetrics`, decremento pessimistico all'invio e
  riconciliazione su `ModifyResponse` / polling periodico `/status`.
- `internal/test/load_balancer_test.go`: suite `TestGeneralLoadBalancer*`.
- `internal/test/resource_test.go`: `TestIndependentResourceChecks`.
"""

import unittest
from dataclasses import replace

from simulator.model.execution.function import Function
from simulator.model.execution.request import Request, RequestStatus
from simulator.model.execution.runtime import BuiltinRuntime
from simulator.model.scheduling.load_balancer import (
    ArchitectureUnawareLoadBalancer, NodeMetric, NodeMetricCache, RandomLoadBalancer,
    RoundRobinLoadBalancer,
)
from simulator.model.topology.infrastructure import Infrastructure
from simulator.model.topology.node import Node


def function(name='hello', archs=('amd64', 'arm64')):
    """Fixture: funzione da 128 MiB e 1 CPU con architetture configurabili."""
    return Function(name, 128, 1, BuiltinRuntime('python314'), supported_archs=archs)


class RoutingTest(unittest.TestCase):
    def test_independent_memory_and_cpu_checks_in_lb_cache(self):
        """Verifica che la cache LB valuti memoria e CPU in modo indipendente e inclusivo al limite.

        Riproduce `TestIndependentResourceChecks` in `internal/test/resource_test.go`.

        - CPU esaurita (`0`) con RAM sufficiente (`256 >= 128`) → non ammissibile.
        - RAM esaurita (`0`) con CPU sufficiente (`2.0 >= 0.5`) → non ammissibile.
        - Capacità esattamente uguale alla domanda (`128 MiB`, `0.5 CPU`) → ammissibile.
        """
        node = Node('resource-check-test', 1024, 2.0, 'amd64')
        fun = Function('test', 128, 0.5, BuiltinRuntime('python314'))
        cache = NodeMetricCache()

        for label, mem, cpu, expected in (
            ('CPU exhausted', 256, 0.0, False),
            ('memory exhausted', 0, 2.0, False),
            ('exact capacity', 128, 0.5, True),
        ):
            with self.subTest(case=label):
                cache._metrics[node.key] = NodeMetric(
                    total_memory_mb=1024, free_memory_mb=mem,
                    total_cpus=2.0, free_cpus=cpu, last_update_s=1.0,
                )
                self.assertEqual(cache.has_enough_resources(node, fun), expected)

    def test_random_balancer_draws_only_compatible_rings_with_reproducible_seed(self):
        """Verifica che `RandomLoadBalancer` selezioni solo ring compatibili con seed riproducibile.

        Riferimento reale: `TestGeneralLoadBalancerTagPattern` e
        `TestGeneralLoadBalancerArchitectureCompatibility` in `internal/test/load_balancer_test.go`.

        - Tra 5 ring (uno vuoto, uno con tag incompatibile, uno con architettura `arm64`, due
          compatibili `amd64` con tag `gpu/nvidia`), la selezione estrae solo `first` e `second`.
        - Lo stesso seed produce sequenze identiche; un seed diverso produce una sequenza diversa.
        """
        infra = Infrastructure()
        for ring_id, arch, tag in (
            ('empty', 'amd64', None),
            ('plain', 'amd64', 'x86/tiny'),
            ('wrong-arch', 'arm64', 'arm/gpu/nvidia'),
            ('first', 'amd64', 'x86-large/gpu/nvidia'),
            ('second', 'amd64', 'x86-tiny/gpu/nvidia'),
        ):
            infra.create_ring(ring_id, arch=arch, memory=512, cpus=2, machine_tag=tag or ring_id)
            if tag is not None:
                infra.create_node(ring_id, ring_id, tag)
        constrained = replace(function(archs=('amd64',)), tag_pattern='GPU, nvidia')

        def draws(seed):
            balancer = RandomLoadBalancer(seed=seed)
            return [
                balancer.submit(Request(constrained), infra.rings.values()).id
                for _ in range(32)
            ]

        baseline = draws(42)
        self.assertEqual(set(baseline), {'first', 'second'})
        self.assertEqual(baseline, draws(42))
        self.assertNotEqual(baseline, draws(43))

    def test_machine_tag_requirements_match_reference_patterns(self):
        """Verifica il matching dei `tag_pattern` (termini, wildcard, regex RE2) come in Go.

        Riferimento reale: `internal/api/api.go` (`normalizeTagPatternForAPI`) e
        `internal/lb/generalLoadBalancer.go` (`matchesMachineTag`).

        - Verifica corrispondenza per termine base, capability, wildcard `*`, regex RE2 (`^...$`),
          requisiti multipli separati da virgola (AND logico) e non corrispondenze.
        - Rifiuta con `ValueError` regex malformate o costrutti PCRE non supportati da RE2
          (lookaround `(?=`, `(?!`, atomic group `(?>`, backreference `\\1`).
        """
        tag = 'x86-large/gpu/nvidia'
        for pattern, expected in (
            ('', True), ('x86', True), ('large', True), ('x86-large/gpu', True),
            ('*/gpu/nvidia', True), ('^x86-.*/gpu.*$', True),
            ('gpu, amd', False), ('x86-tiny/gpu', False),
        ):
            with self.subTest(pattern=pattern):
                self.assertEqual(replace(function(), tag_pattern=pattern).supports_machine_tag(tag), expected)
        base_function = function()
        for bad in ('^[', '^x86(?=/gpu)', '^x86(?!-arm)', '^x86(?>/gpu)$', r'^(x86)-\1$'):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, 'tag_pattern'):
                replace(base_function, tag_pattern=bad)

    def test_fallback_between_compatible_rings_and_rejection_at_saturation(self):
        """Verifica il fallback tra ring compatibili e il rifiuto `out_of_resources` a saturazione.

        Riproduce `TestGeneralLoadBalancerFallbackBetweenCompatiblePools` e
        `TestGeneralLoadBalancerRejectIncompatibleFallback` in `internal/test/load_balancer_test.go`.

        - Tre ring compatibili (`gpu-first`, `gpu-second`, `gpu-third`) e un ring incompatibile (`plain`).
        - Saturato il ring primario, il LB ripiega sul ring compatibile successivo in ordine di
          registrazione; saturati tutti i ring compatibili, rifiuta con `out_of_resources` senza mai
          instradare sul ring incompatibile `plain`.
        """
        seed = 42
        infra = Infrastructure(RandomLoadBalancer(seed=seed))
        nodes = {}
        for ring_id in ('gpu-first', 'gpu-second', 'gpu-third', 'plain'):
            infra.create_ring(ring_id, arch='amd64', memory=128, cpus=1, machine_tag=f'x86/{ring_id}')
            nodes[ring_id] = infra.create_node(ring_id, ring_id, f'x86/{ring_id}', queue_capacity=0)

        fun = replace(function(), tag_pattern='x86/gpu-*')
        requests = (Request(fun), Request(fun))
        predictor = RandomLoadBalancer(seed=seed)
        order = predictor.candidate_rings(requests[0], infra.rings.values())
        primary, fallback, last = (nodes[ring.id] for ring in order)
        fallback_order = [ring_id for ring_id in nodes if ring_id not in (order[0].id, 'plain')]
        self.assertEqual([ring.id for ring in order[1:]], fallback_order)

        self.assertTrue(primary.submit(Request(fun)))
        infra.refresh_load_balancer_metrics()
        self.assertIs(infra.submit(requests[0]), fallback)
        self.assertIs(requests[0].status, RequestStatus.ACCEPTED)

        self.assertTrue(last.submit(Request(fun)))
        infra.refresh_load_balancer_metrics()
        rejected = requests[1]
        self.assertIsNone(infra.submit(rejected))
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(rejected.rejection_reason, 'out_of_resources')
        self.assertEqual(nodes['plain'].resources.used_cpus, 0)

    def test_exact_tag_mask_and_homogeneous_ring_tag_enforcement(self):
        """Verifica il vincolo di tag esatto sul routing e l'omogeneità del `machine_tag` nel ring.

        Riproduce `TestGeneralLoadBalancerExactTagMask` in `internal/test/load_balancer_test.go`.

        - Un nodo con `machine_tag` diverso da quello del ring viene rifiutato con `ValueError`.
        - Una funzione vincolata a `tag_pattern='gpu'` viene instradata solo sul ring `gpu`; una
          funzione con `tag_pattern='arm*'` senza ring compatibili viene rifiutata con `no_destination`,
          lasciando inutilizzato il ring `regular`.
        """
        infra = Infrastructure()
        infra.create_ring('regular', arch='amd64', memory=512, cpus=2)
        regular = infra.create_node('regular', 'one')
        with self.assertRaisesRegex(ValueError, 'machine_tag'):
            infra.create_node('regular', 'three', 'x86/gpu')

        infra.create_ring('gpu', arch='amd64', memory=512, cpus=2, machine_tag='x86/gpu')
        gpu = infra.create_node('gpu', 'three', 'x86/gpu')

        constrained = replace(function(), tag_pattern='gpu')
        self.assertIs(infra.submit(Request(constrained)), gpu)
        rejected = Request(replace(function(), tag_pattern='arm*'))
        self.assertIsNone(infra.submit(rejected))
        self.assertEqual(rejected.rejection_reason, 'no_destination')
        self.assertEqual(regular.resources.used_cpus, 0)

    def test_architecture_metadata_without_machine_tag_routes_normally(self):
        """Verifica il fallback del `machine_tag` sui metadati `arch` quando il tag è omesso.

        Riproduce `TestGeneralLoadBalancerArchitectureMetadataWithoutTag` in
        `internal/test/load_balancer_test.go`.

        - Un ring/nodo registrato con sola architettura (`arm64`) e senza `machine_tag` esplicito
          usa `arm64` come tag di default e riceve regolarmente le invocazioni `arm64`.
        """
        infra = Infrastructure()
        infra.create_ring('arch-only', arch='arm64', memory=512, cpus=2)
        node = infra.create_node('arch-only', 'n-arm')
        self.assertEqual(node.id.machine_tag, 'arm64')

        fun = function(archs=('arm64',))
        self.assertIs(infra.submit(Request(fun)), node)

    def test_balancer_checks_cpu_and_memory_strictly_like_go_proxy(self):
        """Verifica che il LB controlli CPU/RAM disponibili senza considerare `max_concurrency`.

        Riferimento reale: `internal/lb/lb.go` (`DefaultCPUChecker`, `DefaultMemoryChecker`).

        - Con un nodo da 1 CPU e una funzione da 1 CPU e `max_concurrency=2`, la prima invocazione
          consuma la CPU disponibile (`available_cpus=0`).
        - La seconda invocazione via LB viene rifiutata con `out_of_resources` (HTTP 429) perché il
          proxy controlla solo CPU/RAM libere, non gli slot concorrenti del container busy.
        - L'invocazione diretta sul nodo (`node=node`) invece riusa correttamente il container busy.
        """
        infra = Infrastructure()
        infra.create_ring('r', arch='amd64', memory=128, cpus=1)
        node = infra.create_node('r', 'n1')
        fun = Function('mc', 128, 1, BuiltinRuntime('python314'), max_concurrency=2)

        first = Request(fun)
        self.assertIs(infra.submit(first), node)
        node.creation_succeeded(first.container)
        self.assertEqual(node.resources.available_cpus, 0)

        second = Request(fun)
        self.assertIsNone(infra.submit(second))
        self.assertEqual(second.rejection_reason, 'out_of_resources')

        direct = Request(fun)
        self.assertIs(infra.submit(direct, node=node), node)
        self.assertTrue(direct.warm)
        self.assertIs(direct.container, first.container)

    def test_round_robin_balancer_cycles_compatible_rings_and_discovery_order(self):
        """Verifica il ciclo RoundRobin tra ring compatibili e l'ordine di scoperta dei tag.

        Riferimento reale: `internal/lb/generalLoadBalancer.go` (`RoundRobin` con cursore globale
        e aggiunta dei tag all'ordine di scoperta alla registrazione del primo nodo).

        - Parte 1: cicla su `first`, `second`, `third` saltando ring vuoti e incompatibili; una
          richiesta vincolata al solo `third` fa avanzare il cursore globale oltre `third`.
        - Parte 2: creando `a` vuoto e popolando prima `b` e poi `a`, l'ordine di scoperta è `[b, a]`
          e le invocazioni si alternano coerentemente partendo da `b`.
        """
        infra = Infrastructure()
        for ring_id, arch, tag in (
            ('empty', 'amd64', None),
            ('first', 'amd64', 'x86-large/gpu/nvidia'),
            ('wrong-arch', 'arm64', 'arm/gpu/nvidia'),
            ('second', 'amd64', 'x86-tiny/gpu/nvidia'),
            ('third', 'amd64', 'x86-mid/gpu/nvidia'),
        ):
            infra.create_ring(ring_id, arch=arch, memory=512, cpus=2, machine_tag=tag or ring_id)
            if tag is not None:
                infra.create_node(ring_id, ring_id, tag)

        balancer = RoundRobinLoadBalancer()
        all_gpu = replace(function(archs=('amd64',)), tag_pattern='gpu')
        only_third = replace(function(archs=('amd64',)), tag_pattern='x86-mid')

        draws = [balancer.submit(Request(all_gpu), infra.rings.values()).id for _ in range(4)]
        self.assertEqual(draws, ['first', 'second', 'third', 'first'])
        self.assertEqual(balancer.submit(Request(only_third), infra.rings.values()).id, 'third')
        self.assertEqual(balancer.submit(Request(all_gpu), infra.rings.values()).id, 'first')

        # Ordine di scoperta dei tag guidato dalla registrazione del primo nodo.
        infra_rr = Infrastructure(RoundRobinLoadBalancer())
        infra_rr.create_ring('a', arch='amd64', memory=512, cpus=2, machine_tag='amd64/a')
        infra_rr.create_ring('b', arch='amd64', memory=512, cpus=2, machine_tag='amd64/b')
        node_b = infra_rr.create_node('b', 'nb', 'amd64/b')
        self.assertIs(infra_rr.submit(Request(function())), node_b)
        node_a = infra_rr.create_node('a', 'na', 'amd64/a')
        self.assertIs(infra_rr.submit(Request(function())), node_b)
        self.assertIs(infra_rr.submit(Request(function())), node_a)

    def test_architecture_unaware_balancer_uses_single_global_ring_ignoring_machine_tags(self):
        """Verifica `ArchitectureUnawareLoadBalancer`: singolo ring globale, rispetto di `arch` ma non di `tag_pattern`.

        Riferimento reale: `internal/lb/hashRing.go` (`HashRing.Get` filtra per architettura) e
        modalità `ArchitectureUnaware` del LB Go.

        - Una funzione multi-architettura con `tag_pattern='gpu'` ignora il tag e instrada su
          entrambi i nodi (`x86_node`, poi `arm_node`), rifiutando la terza richiesta per saturazione.
        - Una funzione `arm64`-only salta il nodo `amd64` e, dopo la rimozione del nodo `arm64`,
          rifiuta con `no_destination`.
        """
        balancer = ArchitectureUnawareLoadBalancer()
        infra = Infrastructure(balancer)
        infra.create_ring('x86', arch='amd64', memory=128, cpus=1, machine_tag='x86/plain')
        infra.create_ring('arm', arch='arm64', memory=128, cpus=1, machine_tag='arm/gpu')
        x86_node = infra.create_node('x86', 'one', 'x86/plain')
        arm_node = infra.create_node('arm', 'three', 'arm/gpu')

        both = replace(function(archs=('amd64', 'arm64')), tag_pattern='gpu')
        self.assertIs(balancer.select_node(both, infra.rings.values()), x86_node)
        self.assertIs(infra.submit(Request(both)), x86_node)
        self.assertIs(infra.submit(Request(both)), arm_node)
        third = Request(both)
        self.assertIsNone(infra.submit(third))
        self.assertEqual(third.rejection_reason, 'out_of_resources')

        infra_single = Infrastructure(ArchitectureUnawareLoadBalancer())
        infra_single.create_ring('x86', arch='amd64', memory=128, cpus=1)
        infra_single.create_ring('arm', arch='arm64', memory=128, cpus=1)
        infra_single.create_node('x86', 'one')
        arm_only_node = infra_single.create_node('arm', 'three')
        arm_only = replace(function(archs=('arm64',)), tag_pattern='nonexistent-tag')
        self.assertIs(infra_single.submit(Request(arm_only)), arm_only_node)

        infra_single.remove_node('arm', 'three')
        rejected = Request(arm_only)
        self.assertIsNone(infra_single.submit(rejected))
        self.assertEqual(rejected.rejection_reason, 'no_destination')

    def test_empty_and_incompatible_rings_are_skipped(self):
        """Verifica che ring vuoti o di architettura incompatibile non vengano mai selezionati.

        Riproduce `TestGeneralLoadBalancerEmptyRings` e
        `TestGeneralLoadBalancerArchitectureCompatibility` in `internal/test/load_balancer_test.go`.

        - Infrastruttura vuota → `None`.
        - Ring `amd64` vuoto + ring `arm64` popolato con funzione `amd64`-only → `None`, zero CPU usate.
        - Funzione compatibile con `arm64` → instradata correttamente al nodo `arm`.
        """
        infra = Infrastructure()
        self.assertIsNone(infra.submit(Request(function())))
        infra.create_ring('empty', arch='amd64', memory=512, cpus=2)
        infra.create_ring('arm', arch='arm64', memory=512, cpus=2)
        arm = infra.create_node('arm', 'a')
        self.assertIsNone(infra.submit(Request(function(archs=('amd64',)))))
        self.assertEqual(arm.resources.used_cpus, 0)

        request = Request(function())
        self.assertIs(infra.submit(request), arm)
        self.assertIs(request.node, arm)

    def test_hash_lookup_matches_reference_at_boundary_and_wraparound(self):
        """Verifica che il consistent hashing FNV-1a replichi esattamente i boundary e il wraparound di Go.

        Riferimento reale: `internal/lb/hashRing.go` (`hash`, `HashRing.Add`, `HashRing.Get`) e
        `TestGeneralLoadBalancerStableHashWithinPool` in `internal/test/load_balancer_test.go`.

        - Con 2 repliche virtuali per `n1` e `n2`, la chiave esatta `n2#1` cade su `n2`, `hello`
          su `n1` e `foobar` (hash oltre l'ultimo punto) esegue il wraparound sul primo nodo (`n2`).
        """
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2, replicas=2)
        first = infra.create_node('r', 'n1')
        second = infra.create_node('r', 'n2')

        for name, expected in (('n2#1', second), ('hello', first), ('foobar', second)):
            with self.subTest(function=name):
                self.assertIs(ring.select_node(function(name)), expected)

    def test_same_function_uses_reference_order_until_all_nodes_are_saturated(self):
        """Verifica il fallback deterministico intra-ring sull'hash ring fino a saturazione completa.

        Riproduce `TestGeneralLoadBalancerResourceFallbackWithinRing` in
        `internal/test/load_balancer_test.go` con FNV-1a a 128 repliche virtuali.

        - Tre nodi (`one`, `two`, `three`, 2 CPU ciascuno): le invocazioni di `hello` saturano i
          nodi esattamente nell'ordine FNV-1a Go `one → three → two`.
        - Alla settima invocazione (tutti i nodi saturi) il LB rifiuta con `out_of_resources`.
        """
        infra = Infrastructure()
        infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        nodes = {name: infra.create_node('r', name) for name in ('one', 'two', 'three')}

        for name in ('one', 'three', 'two'):
            for _ in range(2):
                request = Request(function())
                self.assertIs(infra.submit(request), nodes[name])
                self.assertIs(request.status, RequestStatus.ACCEPTED)
            self.assertEqual(nodes[name].resources.used_cpus, 2)
            self.assertEqual(nodes[name].resources.busy_pool_used_mem, 256)

        rejected = Request(function())
        self.assertIsNone(infra.submit(rejected))
        self.assertIsNone(rejected.node)
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(rejected.rejection_reason, 'out_of_resources')

    def test_removing_and_readding_nodes_updates_routing_dynamically(self):
        """Verifica aggiunta, rimozione e sostituzione dinamica dei nodi sull'hash ring.

        Riproduce `TestGeneralLoadBalancerAddRemove` in `internal/test/load_balancer_test.go`.

        - La rimozione del primario (`one`) sposta immediatamente il routing su `three` già prima
          della notifica `on_node_removed`; una seconda rimozione dello stesso nodo solleva `ValueError`.
        - Rimosso anche `three`, il ring vuoto rifiuta con `no_destination`.
        - La re-registrazione di un nuovo nodo con nome `one` ripristina il routing sulla nuova istanza.
        """
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        primary = infra.create_node('r', 'one')
        secondary = infra.create_node('r', 'three')
        self.assertIs(ring.select_node(function()), primary)
        observed = []

        def node_removed(infrastructure, target_ring, node):
            self.assertIs(infrastructure.get_ring(target_ring.id), target_ring)
            self.assertNotIn(node, target_ring)
            observed.append((node, target_ring.select_node(function())))

        infra.on_node_removed = node_removed

        self.assertIs(infra.remove_node('r', 'one'), primary)
        with self.assertRaises(ValueError):
            infra.remove_node('r', 'one')
        self.assertIs(infra.submit(Request(function())), secondary)
        self.assertIs(infra.remove_node('r', 'three'), secondary)
        self.assertEqual(observed, [(primary, secondary), (secondary, None)])

        request = Request(function())
        self.assertIsNone(infra.submit(request))
        self.assertEqual(request.rejection_reason, 'no_destination')

        replacement = infra.create_node('r', 'one')
        self.assertIsNot(replacement, primary)
        self.assertIs(infra.submit(Request(function())), replacement)

    def test_resource_cache_pessimistic_decrement_and_reconciliation_match_go_proxy(self):
        """Verifica decremento pessimistico della cache LB e riconciliazione su risposta HTTP.

        Riferimento reale: `internal/lb/lb.go` (`NodeMetrics` decrementate all'invio e riallineate
        in `ModifyResponse` tramite header `Serverledge-Free-Mem`/`Serverledge-Free-CPU`, oltre al
        confronto con la modalità ideale `resource_cache=False`).

        - Con `resource_cache=True`, l'invio decrementa subito `free_memory_mb` e `free_cpus` nella cache.
        - Anche se il worker completa `first`, finché non arriva `reconcile_load_balancer` la cache
          resta stale (`free_cpus=0.0`) e rifiuta `blocked` con `out_of_resources`.
        - Dopo `reconcile_load_balancer`, la cache torna a `(256, 1.0)` e la richiesta successiva viene
          ammessa con riuso warm.
        - Con `resource_cache=False`, il controllo legge lo stato istantaneo del worker e accetta subito.
        """
        lb = RandomLoadBalancer(seed=1, resource_cache=True, refresh_interval_s=2.0)
        infra = Infrastructure(lb)
        infra.create_ring('r', arch='amd64', memory=256, cpus=1.0)
        node = infra.create_node('r', 'n1')
        fun = Function('f', 128, 1.0, BuiltinRuntime('python314'))

        first = Request(fun)
        self.assertIs(infra.submit(first, now=1.0), node)
        node.creation_succeeded(first.container)
        metric = lb._metric_cache.get(node)
        self.assertIsNotNone(metric)
        self.assertEqual((metric.free_memory_mb, metric.free_cpus), (128, 0.0))

        node.complete(first.container, 1.5)
        self.assertEqual(lb._metric_cache.get(node).free_cpus, 0.0)
        blocked = Request(fun)
        self.assertIsNone(infra.submit(blocked, now=1.5))
        self.assertEqual(blocked.rejection_reason, 'out_of_resources')

        infra.reconcile_load_balancer(first, now=1.5)
        metric = lb._metric_cache.get(node)
        self.assertEqual((metric.free_memory_mb, metric.free_cpus), (256, 1.0))
        second = Request(fun)
        self.assertIs(infra.submit(second, now=1.6), node)
        self.assertTrue(second.warm)

        lb_ideal = RandomLoadBalancer(seed=1, resource_cache=False)
        infra_ideal = Infrastructure(lb_ideal)
        infra_ideal.create_ring('r', arch='amd64', memory=256, cpus=1.0)
        node_ideal = infra_ideal.create_node('r', 'n1')
        r1 = Request(fun)
        self.assertIs(infra_ideal.submit(r1, now=1.0), node_ideal)
        node_ideal.creation_succeeded(r1.container)
        node_ideal.complete(r1.container, 1.5)
        r2 = Request(fun)
        self.assertIs(infra_ideal.submit(r2, now=1.5), node_ideal)

    def test_scenario_config_defaults_to_real_resource_cache_and_allows_opt_out(self):
        """Verifica che lo scenario attivi di default la cache risorse reale (30s) con opt-out esplicito.

        Riferimento reale: `internal/lb/lb.go` (`refresh_interval=30s` di default nel proxy Go).

        - Senza opzioni esplicite, `Scenario` configura `load_balancer_resource_cache=True` e
          `load_balancer_refresh_interval_s=30.0`, propagati al `LoadBalancer` della regione.
        - Impostando `load_balancer_resource_cache=False` e `load_balancer_refresh_interval_s=5.0`,
          entrambi i parametri vengono propagati al `LoadBalancer`.
        """
        from simulator.core.simulation import Simulation
        from simulator.io.scenario import Scenario
        from simulator.model.execution.function import FunctionProfile

        fn = Function(
            'hello', 128, 1.0, BuiltinRuntime('python314'),
            profile=FunctionProfile(arrival_rate=1.0, service_rate=2.0, first_arrival_time=0.0),
        )
        infra_cfg = {'rings': [{'id': 'r', 'arch': 'amd64', 'memory_mb': 256, 'cpus': 2.0, 'nodes': [{'name': 'n1'}]}]}

        default_scenario = Scenario((fn,), infrastructure=infra_cfg)
        self.assertTrue(default_scenario.load_balancer_resource_cache)
        self.assertEqual(default_scenario.load_balancer_refresh_interval_s, 30.0)
        lb_default = Simulation(default_scenario).regions['default'].infrastructure.load_balancer
        self.assertTrue(lb_default.resource_cache_enabled)

        ideal_scenario = Scenario(
            (fn,), infrastructure=infra_cfg,
            load_balancer_resource_cache=False, load_balancer_refresh_interval_s=5.0,
        )
        self.assertFalse(ideal_scenario.load_balancer_resource_cache)
        self.assertEqual(ideal_scenario.load_balancer_refresh_interval_s, 5.0)
        lb_ideal = Simulation(ideal_scenario).regions['default'].infrastructure.load_balancer
        self.assertFalse(lb_ideal.resource_cache_enabled)
        self.assertEqual(lb_ideal.refresh_interval_s, 5.0)


if __name__ == '__main__':
    unittest.main()
