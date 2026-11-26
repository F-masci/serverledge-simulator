"""Percorso di inoltro iniziale e compatibilità dell'hashing.

Riferimenti Infrascaler: internal/lb/hashRing.go, hash, HashRing.Add e
HashRing.Get per l'ordine delle repliche virtuali e la ricerca di altri nodi;
internal/lb/generalLoadBalancer.go per il filtro di compatibilità e il fallback.
"""

import unittest
from dataclasses import replace

from simulator.model.execution.function import Function
from simulator.model.execution.request import Request, RequestStatus
from simulator.model.execution.runtime import BuiltinRuntime
from simulator.model.scheduling.load_balancer import (
    ArchitectureUnawareLoadBalancer, MABLoadBalancer, RandomLoadBalancer, RoundRobinLoadBalancer,
)
from simulator.model.topology.infrastructure import Infrastructure


def function(name='hello', archs=('amd64', 'arm64')):
    return Function(name, 128, 1, BuiltinRuntime('python314'), supported_archs=archs)


class RoutingTest(unittest.TestCase):
    def test_random_balancer_draws_only_compatible_rings_with_reproducible_seed(self):
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

    def test_fallback_uses_registration_order_and_rejects_after_saturation(self):
        seed = 42
        infra = Infrastructure(RandomLoadBalancer(seed=seed))
        nodes = {}
        for ring_id in ('gpu-first', 'gpu-second', 'gpu-third', 'plain'):
            infra.create_ring(ring_id, arch='amd64', memory=128, cpus=1, machine_tag=f'x86/{ring_id}')
            nodes[ring_id] = infra.create_node(
                ring_id, ring_id, f'x86/{ring_id}', queue_capacity=0,
            )

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
        payload = rejected.to_dict()
        self.assertIsNone(payload['node_id'])
        self.assertIsNone(payload['container_id'])
        self.assertEqual(nodes['plain'].resources.used_cpus, 0)

    def test_ring_enforces_homogeneous_machine_tag_and_routes_by_ring_tag(self):
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

    def test_balancer_checks_cpu_and_memory_strictly_like_go_proxy(self):
        infra = Infrastructure()
        infra.create_ring('r', arch='amd64', memory=128, cpus=1)
        node = infra.create_node('r', 'n1')
        fun = Function('mc', 128, 1, BuiltinRuntime('python314'), max_concurrency=2)

        first = Request(fun)
        self.assertIs(infra.submit(first), node)
        node.creation_succeeded(first.container)
        self.assertEqual(node.resources.available_cpus, 0)

        # Il proxy Go controlla solo CPU/RAM disponibili (non gli slot busy di max_concurrency).
        second = Request(fun)
        self.assertIsNone(infra.submit(second))
        self.assertEqual(second.rejection_reason, 'out_of_resources')
        # L'invocazione diretta sul nodo invece riusa il container busy.
        direct = Request(fun)
        self.assertIs(infra.submit(direct, node=node), node)
        self.assertTrue(direct.warm)
        self.assertIs(direct.container, first.container)

    def test_round_robin_balancer_cycles_compatible_rings_in_registration_order(self):
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

        # Cicla sugli anelli compatibili saltando quelli vuoti o con architettura incompatibile.
        draws = [
            balancer.submit(Request(all_gpu), infra.rings.values()).id
            for _ in range(4)
        ]
        self.assertEqual(draws, ['first', 'second', 'third', 'first'])

        # Una richiesta compatibile solo con 'third' avanza l'indice globale oltre 'third',
        # quindi la richiesta successiva riparte da 'first'.
        self.assertEqual(balancer.submit(Request(only_third), infra.rings.values()).id, 'third')
        self.assertEqual(balancer.submit(Request(all_gpu), infra.rings.values()).id, 'first')

    def test_round_robin_discovers_new_tag_on_first_node_like_go(self):
        infra = Infrastructure(RoundRobinLoadBalancer())
        infra.create_ring('a', arch='amd64', memory=512, cpus=2, machine_tag='amd64/a')
        infra.create_ring('b', arch='amd64', memory=512, cpus=2, machine_tag='amd64/b')
        node_b = infra.create_node('b', 'nb', 'amd64/b')

        # Con 'a' vuoto e 'b' attivo, la prima richiesta sceglie 'b' e _rr_index torna a 0.
        self.assertIs(infra.submit(Request(function())), node_b)

        # Aggiungendo il primo nodo ad 'a', 'a' entra in coda a [b, a]: la richiesta successiva sceglie ancora 'b'.
        node_a = infra.create_node('a', 'na', 'amd64/a')
        self.assertIs(infra.submit(Request(function())), node_b)
        self.assertIs(infra.submit(Request(function())), node_a)

    def test_round_robin_uses_eager_node_activation_order_before_first_request(self):
        infra = Infrastructure(RoundRobinLoadBalancer())
        infra.create_ring('a', arch='amd64', memory=512, cpus=2, machine_tag='amd64/a')
        infra.create_ring('b', arch='amd64', memory=512, cpus=2, machine_tag='amd64/b')
        node_b = infra.create_node('b', 'nb', 'amd64/b')
        node_a = infra.create_node('a', 'na', 'amd64/a')
        with self.assertRaisesRegex(ValueError, 'già presente'):
            infra.create_node('a', 'nb', 'amd64/a')

        # I nodi sono stati registrati in ordine b poi a: la prima richiesta deve scegliere b.
        self.assertIs(infra.submit(Request(function())), node_b)
        self.assertIs(infra.submit(Request(function())), node_a)

    def test_unavailable_balancers_fail_on_instantiation(self):
        with self.assertRaisesRegex(NotImplementedError, 'MAB'):
            MABLoadBalancer()

    def test_architecture_unaware_balancer_uses_single_global_ring_ignoring_machine_tags(self):
        balancer = ArchitectureUnawareLoadBalancer()
        infra = Infrastructure(balancer)
        infra.create_ring('x86', arch='amd64', memory=128, cpus=1, machine_tag='x86/plain')
        infra.create_ring('arm', arch='arm64', memory=128, cpus=1, machine_tag='arm/gpu')
        x86_node = infra.create_node('x86', 'one', 'x86/plain')
        arm_node = infra.create_node('arm', 'three', 'arm/gpu')

        # Ignora tag_pattern, ma rispetta comunque supported_archs come HashRing.Get in Go.
        both = replace(function(archs=('amd64', 'arm64')), tag_pattern='gpu')
        first_expected = balancer.select_node(both, infra.rings.values())
        self.assertIs(first_expected, x86_node)

        first = Request(both)
        self.assertIs(infra.submit(first), x86_node)
        second = Request(both)
        self.assertIs(infra.submit(second), arm_node)
        third = Request(both)
        self.assertIsNone(infra.submit(third))
        self.assertEqual(third.rejection_reason, 'out_of_resources')

        # Se la funzione supporta una sola architettura, salta i nodi incompatibili.
        infra_single = Infrastructure(ArchitectureUnawareLoadBalancer())
        infra_single.create_ring('x86', arch='amd64', memory=128, cpus=1)
        infra_single.create_ring('arm', arch='arm64', memory=128, cpus=1)
        infra_single.create_node('x86', 'one')
        arm_only_node = infra_single.create_node('arm', 'three')
        arm_only = replace(function(archs=('arm64',)), tag_pattern='nonexistent-tag')
        self.assertIs(infra_single.submit(Request(arm_only)), arm_only_node)

        # La rimozione di un nodo aggiorna il ring globale.
        infra_single.remove_node('arm', 'three')
        rejected = Request(arm_only)
        self.assertIsNone(infra_single.submit(rejected))
        self.assertEqual(rejected.rejection_reason, 'no_destination')

    def test_empty_and_incompatible_rings_are_skipped(self):
        """Non si inoltra a ring vuoti o non compatibili; senza destinazione restituisce None."""
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
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2, replicas=2)
        first = infra.create_node('r', 'n1')
        second = infra.create_node('r', 'n2')

        # Go HashRing.Add/Get: n2#1 è l'ultimo punto di n2, seguito da n1.
        # L'hash di foobar supera tutti i punti e riparte dal primo (n2).
        for name, expected in (('n2#1', second), ('hello', first), ('foobar', second)):
            with self.subTest(function=name):
                self.assertIs(ring.select_node(function(name)), expected)

    def test_same_function_uses_reference_order_until_all_nodes_are_saturated(self):
        infra = Infrastructure()
        infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        nodes = {name: infra.create_node('r', name) for name in ('one', 'two', 'three')}

        # Ordine ottenuto con FNV-1a e 128 repliche in Go, saturando un nodo alla volta.
        for name in ('one', 'three', 'two'):
            for _ in range(2):
                request = Request(function())
                self.assertIs(infra.submit(request), nodes[name])
                self.assertIs(request.node, nodes[name])
                self.assertIs(request.status, RequestStatus.ACCEPTED)
            self.assertEqual(nodes[name].resources.used_cpus, 2)
            self.assertEqual(nodes[name].resources.busy_pool_used_mem, 256)

        rejected = Request(function())
        self.assertIsNone(infra.submit(rejected))
        self.assertIsNone(rejected.node)
        self.assertIs(rejected.status, RequestStatus.REJECTED)
        self.assertEqual(rejected.rejection_reason, 'out_of_resources')

    def test_removing_primary_updates_routing_before_notification(self):
        infra = Infrastructure()
        ring = infra.create_ring('r', arch='amd64', memory=512, cpus=2)
        primary = infra.create_node('r', 'one')
        secondary = infra.create_node('r', 'three')
        self.assertIs(ring.select_node(function()), primary)
        observed = []

        def node_removed(infrastructure, ring, node):
            self.assertIs(infrastructure.get_ring(ring.id), ring)
            self.assertNotIn(node, ring)
            observed.append((node, ring.select_node(function())))

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
        # Con resource_cache=True (default), il LB decrementa la propria cache a ogni invio
        # e la riconcilia solo su reconcile_request (ModifyResponse) o refresh_metrics (/status).
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

        # Completa first sul worker: il nodo torna subito libero (1.0 CPU, 256 MB reclamabili),
        # ma finché non torna la risposta al proxy (ModifyResponse) la cache LB resta stale a 0.0 CPU
        # e rifiuta il nuovo arrivo con out_of_resources.
        node.complete(first.container, 1.5)
        self.assertEqual(lb._metric_cache.get(node).free_cpus, 0.0)
        blocked = Request(fun)
        self.assertIsNone(infra.submit(blocked, now=1.5))
        self.assertEqual(blocked.rejection_reason, 'out_of_resources')

        # Al ritorno della risposta (reconcile_load_balancer), la cache viene riconciliata
        # allo stato reale del nodo e il proxy torna ad accettare richieste.
        infra.reconcile_load_balancer(first, now=1.5)
        metric = lb._metric_cache.get(node)
        self.assertEqual((metric.free_memory_mb, metric.free_cpus), (256, 1.0))
        second = Request(fun)
        self.assertIs(infra.submit(second, now=1.6), node)
        self.assertTrue(second.warm)

        # Con resource_cache=False (modalità ideale precedente), il controllo legge subito
        # lo stato istantaneo del nodo e accetta senza attendere riconciliazione.
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
        sim_default = Simulation(default_scenario)
        lb_default = sim_default.regions['default'].infrastructure.load_balancer
        self.assertTrue(lb_default.resource_cache_enabled)

        ideal_scenario = Scenario(
            (fn,), infrastructure=infra_cfg,
            load_balancer_resource_cache=False, load_balancer_refresh_interval_s=5.0,
        )
        self.assertFalse(ideal_scenario.load_balancer_resource_cache)
        self.assertEqual(ideal_scenario.load_balancer_refresh_interval_s, 5.0)
        sim_ideal = Simulation(ideal_scenario)
        lb_ideal = sim_ideal.regions['default'].infrastructure.load_balancer
        self.assertFalse(lb_ideal.resource_cache_enabled)
        self.assertEqual(lb_ideal.refresh_interval_s, 5.0)


if __name__ == '__main__':
    unittest.main()
