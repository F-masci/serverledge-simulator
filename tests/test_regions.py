"""Test di aderenza dell'offloading interregionale e isolamento tra regioni rispetto a Serverledge Go.

Riferimenti al sistema reale (`serverledge-infrascaler`):
- `internal/scheduling/offloading.go`: offloading verso regione cloud remota con flag
  `CanDoOffloading=false` per impedire il multi-hop tra regioni.
- `internal/scheduling/cloudonly_policy.go` e `internal/scheduling/edgeOnlyPolicy.go`:
  separazione tra vicini intra-regione e destinazione cloud interregionale.
"""

import unittest

from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario
from simulator.model.execution.request import RequestStatus
from simulator.observer import NotificationType, observer

FUNCTION = {
    'name': 'f', 'runtime': 'python314', 'memory_mb': 128, 'cpu_demand': 1,
    'arrival_rate': 0.1, 'service_rate': 0.5, 'first_arrival_time': 0,
    'arrival_distribution': 'deterministic', 'service_distribution': 'deterministic',
}


def simulated_region(*, latency=0, **node_settings):
    return {
        'offload_latency_s': latency,
        'infrastructure': {'rings': [{
            'id': 'amd64', 'arch': 'amd64', 'memory_mb': 128, 'cpus': 1,
            'nodes': [{'name': 'worker', **node_settings}],
        }]},
    }


class RegionsTest(unittest.TestCase):
    def setUp(self):
        self.notifications = []
        self.addCleanup(observer.subscribe(self.notifications.append))

    def advance_to(self, simulation, time):
        engine = simulation._ctx.engine
        while engine.now < time:
            self.assertTrue(engine.step())
        self.assertEqual(engine.now, time)

    def test_offload_to_always_successful_region_counts_one_arrival_and_round_trip(self):
        """Verifica l'offload verso una regione cloud sintetica (`always_succeeds`) con RTT esplicito.

        - Una richiesta in arrivo su `edge` (policy `cloudonly`) viene inoltrata a `cloud`
          (`offload_latency_by_source_s={'edge': 0.25}` → 0.125s andata + 0.375s servizio + 0.125s ritorno).
        - Conta un solo arrivo esterno su `edge`, `in_transit=1` durante il transito, ammissione senza
          container a `t=0.125` e completamento a `t=0.625` (`mean_response_time_s=0.625`).
        - La sequenza di notifiche rispetta l'ordine causale completo senza nodi o container su `cloud`.
        """
        simulation = Simulation(Scenario(
            [dict(FUNCTION, arrival_rate=1)], invocation={'can_offload': True}, regions={
                'edge': simulated_region(policy='cloudonly', remote_region='cloud'),
                'cloud': {
                    'mode': 'always_succeeds', 'service_time_s': 0.375,
                    'offload_latency_s': 0.5, 'offload_latency_by_source_s': {'edge': 0.25},
                },
            },
        ))

        initial = simulation.run(arrivals=1).metrics
        self.assertEqual(initial['primary_region'], 'edge')
        self.assertEqual(initial['total']['arrivals'], 1)
        self.assertEqual(initial['total']['in_transit'], 1)
        self.assertEqual(initial['regions']['edge']['external_arrivals'], 1)
        self.assertEqual(initial['regional_offloads'], [{'source': 'edge', 'destination': 'cloud', 'count': 1}])
        self.assertIsNone(simulation.regions['cloud'].infrastructure)

        self.advance_to(simulation, 0.125)
        accepted = simulation.metrics_snapshot()['total']
        self.assertEqual((accepted['arrivals'], accepted['accepted'], accepted['in_transit']), (1, 1, 0))
        self.assertEqual(accepted['accepted_without_container'], 1)

        self.advance_to(simulation, 0.625)
        snapshot = simulation.metrics_snapshot()
        self.assertEqual(snapshot['total']['completed'], 1)
        self.assertEqual(snapshot['total']['mean_response_time_s'], 0.625)
        self.assertEqual(snapshot['total']['mean_service_time_s'], 0.375)
        self.assertEqual(snapshot['total']['warm_starts'] + snapshot['total']['cold_starts'], 0)
        self.assertEqual(snapshot['regions']['cloud']['invocations']['completed'], 1)

        registered, arrived, forwarded, delivered, admitted, completed = self.notifications
        self.assertEqual([event.type for event in self.notifications], [
            NotificationType.NODE_REGISTERED, NotificationType.REQUEST_ARRIVED,
            NotificationType.REQUEST_FORWARDED, NotificationType.REQUEST_DELIVERED,
            NotificationType.REQUEST_ACCEPTED, NotificationType.REQUEST_COMPLETED,
        ])
        edge, cloud = simulation.regions['edge'], simulation.regions['cloud']
        self.assertIs(registered.region, edge)
        self.assertEqual((forwarded.source_region, forwarded.region, delivered.sim_time_s), (edge, cloud, 0.125))
        self.assertEqual((arrived.region, completed.region), (edge, cloud))
        self.assertIsNone(completed.node)
        self.assertIsNone(completed.container)

    def test_offload_to_simulated_region_releases_remote_resources_before_return(self):
        """Verifica che su offload verso regione simulata le risorse remote vengano rilasciate prima del ritorno.

        - Con RTT=4s (2s andata, 2s servizio, 2s ritorno), a `t=2` il nodo su `cloud` occupa 1 CPU
          mentre `edge` resta a 0 CPU.
        - A `t=4` il servizio su `cloud` termina e rilascia subito la CPU remota (`used_cpus=0`),
          mentre la richiesta resta `pending=1` finché la risposta non rientra al client a `t=6`.
        """
        scenario = Scenario([FUNCTION], primary_region='edge', invocation={'can_offload': True}, regions={
            'edge': simulated_region(policy='cloudonly', remote_region='cloud'),
            'cloud': simulated_region(latency=4),
        })
        simulation = Simulation(scenario)
        simulation.run(arrivals=1)

        self.advance_to(simulation, 2)
        snapshot = simulation.metrics_snapshot()
        self.assertEqual(snapshot['nodes_by_ring']['amd64']['worker']['used_cpus'], 0)
        self.assertEqual(snapshot['regions']['cloud']['nodes_by_ring']['amd64']['worker']['used_cpus'], 1)

        self.advance_to(simulation, 4)
        snapshot = simulation.metrics_snapshot()
        self.assertEqual(snapshot['regions']['cloud']['nodes_by_ring']['amd64']['worker']['used_cpus'], 0)
        self.assertEqual((snapshot['total']['completed'], snapshot['total']['pending']), (0, 1))

        self.advance_to(simulation, 6)
        final = simulation.metrics_snapshot()['total']
        self.assertEqual((final['arrivals'], final['completed'], final['pending']), (1, 1, 0))
        self.assertEqual(final['mean_response_time_s'], 6)
        self.assertEqual(final['cold_starts'], 1)

    def test_offload_uses_source_specific_round_trip_or_destination_default(self):
        """Verifica la risoluzione della latenza RTT per coppia `(sorgente, destinazione)` con fallback al default.

        - `north` usa l'override `2s` (ammissione a `t=1`, completamento a `t=2.5`).
        - `local` usa l'override `0s` (ammissione immediata a `t=0`, completamento a `t=0.5`).
        - `south` non ha override e ricade su `offload_latency_s=4s` (ammissione a `t=2`, completamento a `t=4.5`).
        """
        sources = ('north', 'local', 'south')
        regions = {source: simulated_region(policy='cloudonly', remote_region='cloud') for source in sources}
        regions['cloud'] = {
            'mode': 'always_succeeds',
            'service_time_s': 0.5,
            'offload_latency_s': 4,
            'offload_latency_by_source_s': {'north': 2, 'local': 0},
        }
        for source, admission, completion in (
            ('north', 1, 2.5), ('local', 0, 0.5), ('south', 2, 4.5),
        ):
            with self.subTest(source=source):
                simulation = Simulation(Scenario(
                    [FUNCTION], regions=regions,
                    invocation={'entry_region': source, 'can_offload': True},
                ))
                simulation.run(arrivals=1)
                self.advance_to(simulation, completion)
                received, = [event for event in self.notifications
                             if event.run_id == simulation.run_id
                             and event.type is NotificationType.REQUEST_ACCEPTED]
                completed, = [event for event in self.notifications
                              if event.run_id == simulation.run_id
                              and event.type is NotificationType.REQUEST_COMPLETED]
                self.assertEqual(received.sim_time_s, admission)
                self.assertEqual(completed.response_time_s, completion)

    def test_cannot_offload_again_from_a_simulated_destination(self):
        """Verifica il divieto di multi-hop interregionale (`can_offload=False` dopo il primo offload).

        Riferimento reale: `internal/scheduling/offloading.go` (la richiesta inoltrata verso una
        regione remota viaggia con `CanDoOffloading=false`).

        - Catena `edge → relay → cloud` con policy `cloudonly` su `edge` e `relay`.
        - La richiesta arrivata su `relay` da `edge` ha `can_offload=False` e viene rifiutata
          invece di essere ulteriormente inoltrata a `cloud`.
        """
        scenario = Scenario([FUNCTION], invocation={'can_offload': True}, regions={
            'edge': simulated_region(policy='cloudonly', remote_region='relay'),
            'relay': simulated_region(policy='cloudonly', remote_region='cloud'),
            'cloud': {'mode': 'always_succeeds'},
        })
        simulation = Simulation(scenario)
        requests = []
        simulation.regions['relay'].infrastructure.on_request_submitted = (
            lambda _, request, node: requests.append(request)
        )
        stats = simulation.run(arrivals=1).metrics
        request, = requests
        self.assertIs(request.status, RequestStatus.REJECTED)
        self.assertFalse(request.can_offload)
        self.assertEqual((request.to_dict()['entry_region'], request.to_dict()['region']), ('edge', 'relay'))
        self.assertEqual(stats['total']['rejected'], 1)
        self.assertEqual(stats['regional_offloads'], [{'source': 'edge', 'destination': 'relay', 'count': 1}])

    def test_remote_rejection_pays_return_latency(self):
        """Verifica che un rifiuto nella regione remota paghi la latenza di ritorno verso il client.

        - Con `RTT=4s` tra `edge` e `relay`, alla consegna (`t=2`) `relay` rifiuta la richiesta:
          il rifiuto resta in transito (`in_transit=1`, `pending=1`, `rejected=0`) fino a `t=4`,
          istante in cui viene notificato `REQUEST_REJECTED`.
        """
        scenario = Scenario([FUNCTION], invocation={'can_offload': True}, regions={
            'edge': simulated_region(policy='cloudonly', remote_region='relay'),
            'relay': simulated_region(latency=4, policy='cloudonly', remote_region='cloud'),
            'cloud': {'mode': 'always_succeeds'},
        })
        simulation = Simulation(scenario)
        simulation.run(arrivals=1)

        self.advance_to(simulation, 2)
        mid = simulation.metrics_snapshot()['total']
        self.assertEqual((mid['arrivals'], mid['rejected'], mid['in_transit'], mid['pending']), (1, 0, 1, 1))

        self.advance_to(simulation, 4)
        final = simulation.metrics_snapshot()['total']
        self.assertEqual((final['arrivals'], final['rejected'], final['in_transit'], final['pending']), (1, 1, 0, 0))
        rejected, = [
            event for event in self.notifications
            if event.run_id == simulation.run_id and event.type is NotificationType.REQUEST_REJECTED
        ]
        self.assertEqual(rejected.sim_time_s, 4)

    def test_remote_region_uses_the_balancer_or_checks_the_registered_node(self):
        """Verifica la differenza tra regione remota con LB (`has_load_balancer=True`) e senza LB.

        - La regione `cloud` espone un primo nodo `arm64` e un secondo nodo `amd64`; la funzione
          supporta solo `amd64`.
        - Con `has_load_balancer=True`, il LB remoto seleziona il nodo `amd64` e accetta la richiesta.
        - Con `has_load_balancer=False`, `edge` conosce solo il nodo remoto registrato (`arm64`) e
          rifiuta localmente prima dell'inoltro perché incompatibile.
        """
        cloud = simulated_region()
        cloud['infrastructure']['rings'].insert(0, {
            'id': 'arm', 'arch': 'arm64', 'machine_tag': 'a', 'memory_mb': 128, 'cpus': 1,
            'nodes': [{'name': 'first'}],
        })
        function = dict(FUNCTION, supported_archs=['amd64'])
        for enabled in (False, True):
            with self.subTest(has_load_balancer=enabled):
                scenario = Scenario([function], invocation={'can_offload': True}, regions={
                    'edge': simulated_region(policy='cloudonly', remote_region='cloud'),
                    'cloud': dict(cloud, has_load_balancer=enabled),
                })
                stats = Simulation(scenario).run(arrivals=1).metrics
                self.assertEqual(stats['regions']['cloud']['invocations']['accepted'], int(enabled))
                self.assertEqual(stats['regions']['edge']['invocations']['rejected'], int(not enabled))
                self.assertEqual(len(stats['regional_offloads']), int(enabled))

    def test_entry_region_can_override_primary_and_complete_immediately(self):
        """Verifica che `invocation.entry_region` instradi gli arrivi direttamente sulla regione indicata.

        - Con `primary_region='edge'` ma `entry_region='cloud'` (`always_succeeds`, latenza 0),
          l'arrivo viene conteggiato come esterno su `cloud` e completa a `t=0` senza offload.
        """
        scenario = Scenario([FUNCTION], primary_region='edge', regions={
            'edge': simulated_region(),
            'cloud': {'mode': 'always_succeeds'},
        }, invocation={'entry_region': 'cloud'})
        stats = Simulation(scenario).run(arrivals=1).metrics
        self.assertEqual((stats['total']['arrivals'], stats['total']['completed']), (1, 1))
        self.assertEqual(stats['total']['mean_response_time_s'], 0)
        self.assertEqual(stats['regions']['cloud']['external_arrivals'], 1)
        self.assertEqual(stats['regions']['edge']['external_arrivals'], 0)
        self.assertEqual(stats['regional_offloads'], [])

    def test_routing_and_edge_neighbors_stay_in_their_region(self):
        """Verifica che i vicini `edgeonly` siano strettamente confinati alla regione di appartenenza.

        Riferimento reale: `internal/scheduling/edgeOnlyPolicy.go` (discovery etcd per singola area).

        - Un nodo `edgeonly` senza altri nodi nella propria regione `edge` non vede i nodi presenti
          nella regione `other` e rifiuta la richiesta senza generare arrivi su `other`.
        """
        scenario = Scenario([dict(FUNCTION, first_arrival_time=2)], regions={
            'edge': simulated_region(policy='edgeonly'),
            'other': simulated_region(),
        }, invocation={'can_offload': True})
        stats = Simulation(scenario).run(arrivals=1).metrics
        self.assertEqual(stats['regions']['edge']['invocations']['rejected'], 1)
        self.assertEqual(stats['regions']['other']['invocations']['arrivals'], 0)


if __name__ == '__main__':
    unittest.main()
