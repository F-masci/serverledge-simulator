"""Test di riproducibilità, isolamento dei flussi PRNG e correttezza statistica delle distribuzioni.

Verifica il contratto di `RandomStreams` (PCG64DXSM + `SeedSequence` deterministico
indipendente da `PYTHONHASHSEED`) e i momenti dei processi di arrivo (Poisson) e
durata di servizio (Gamma).
"""

import os
import subprocess
import sys
import unittest

import numpy as np
from simulator.random import GammaServiceTime, PoissonArrivalProcess, RandomStreams, StreamKind


class RandomStreamsTest(unittest.TestCase):
    def test_repeated_lookup_continues_the_same_stream(self):
        """Verifica che richieste ripetute della stessa chiave `(kind, id)` avanzino lo stesso generatore.

        - Due chiamate consecutive a `stream(ARRIVAL, 'f/main')` restituiscono la medesima istanza
          e la concatenazione di due blocchi da 4 campioni coincide con un blocco unico da 8 campioni.
        """
        streams = RandomStreams(1234)
        first = streams.stream(StreamKind.ARRIVAL, 'f/main')
        a = first.random(4)
        again = streams.stream(StreamKind.ARRIVAL, 'f/main')
        self.assertIs(first, again)
        b = again.random(4)
        reference = RandomStreams(1234).stream(StreamKind.ARRIVAL, 'f/main')
        np.testing.assert_array_equal(np.concatenate((a, b)), reference.random(8))

    def test_creation_order_and_extra_draws_do_not_change_other_streams(self):
        """Verifica il Common Random Numbers (CRN): estrazioni su altri flussi non perturbano il flusso `SERVICE/f`.

        - Consumare 1000 campioni su `ARRIVAL/f` e creare `SERVICE/unrelated` lascia identica la
          sequenza prodotta da `SERVICE/f` rispetto al baseline.
        """
        baseline = RandomStreams(42)
        expected = baseline.stream(StreamKind.SERVICE, 'f').random(20)
        changed = RandomStreams(42)
        changed.stream(StreamKind.ARRIVAL, 'f').random(1000)
        changed.stream(StreamKind.SERVICE, 'unrelated').random(30)
        np.testing.assert_array_equal(expected, changed.stream(StreamKind.SERVICE, 'f').random(20))

    def test_purpose_identity_and_replication_select_distinct_sequences(self):
        """Verifica che scopo (`StreamKind`), identificatore e indice di replica generino flussi distinti.

        - Confronta tutte le coppie tra `(0, ARRIVAL, f)`, `(0, SERVICE, f)`, `(0, ARRIVAL, g)` e
          `(1, ARRIVAL, f)` assicurando che nessuna coppia produca la stessa sequenza.
        """
        sequences = [
            RandomStreams(42, replica).stream(kind, name).random(10)
            for replica, kind, name in (
                (0, StreamKind.ARRIVAL, 'f'), (0, StreamKind.SERVICE, 'f'),
                (0, StreamKind.ARRIVAL, 'g'), (1, StreamKind.ARRIVAL, 'f'),
            )
        ]
        for index, a in enumerate(sequences):
            for b in sequences[index + 1:]:
                self.assertFalse(np.array_equal(a, b))

    def test_reproducible_across_processes_and_python_hash_seeds(self):
        """Verifica la riproducibilità cross-processo indipendentemente da `PYTHONHASHSEED`.

        - Avvia due sottoprocessi Python con `PYTHONHASHSEED=1` e `PYTHONHASHSEED=999` e verifica
          che entrambi estraggano la medesima sequenza per `RandomStreams(42).stream(SERVICE, 'f')`.
        """
        code = (
            'import json; from simulator.random import RandomStreams, StreamKind; '
            'print(json.dumps(RandomStreams(42).stream(StreamKind.SERVICE, "f").random(5).tolist()))'
        )
        outputs = []
        for hash_seed in ('1', '999'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            outputs.append(subprocess.check_output([sys.executable, '-c', code], env=env, text=True))
        self.assertEqual(outputs[0], outputs[1])


class DistributionsTest(unittest.TestCase):
    def test_poisson_interarrivals_have_expected_mean(self):
        """Verifica che `PoissonArrivalProcess(rate=4)` produca interarrivi esponenziali non negativi con media ~0.25s."""
        process = PoissonArrivalProcess(4, RandomStreams(10), 'f/main')
        samples = np.array([process.next_iat() for _ in range(30000)])
        self.assertTrue(np.all(samples >= 0))
        self.assertAlmostEqual(samples.mean(), 0.25, delta=0.01)

    def test_gamma_service_has_expected_mean_and_scv(self):
        """Verifica che `GammaServiceTime(mean=2, scv=0.5)` riproduca media e coefficiente di variazione quadrato attesi."""
        service = GammaServiceTime(2, 0.5, RandomStreams(10), 'f')
        samples = np.array([service.sample() for _ in range(30000)])
        self.assertTrue(np.all(samples >= 0))
        self.assertAlmostEqual(samples.mean(), 2, delta=0.05)
        self.assertAlmostEqual(samples.var() / samples.mean() ** 2, 0.5, delta=0.04)

    def test_distribution_samples_reproduce_after_reconstruction(self):
        """Verifica che ricostruire `RandomStreams(99, 2)` e i campionatori produca la stessa sequenza esatta."""
        def sample():
            streams = RandomStreams(99, 2)
            arrivals = PoissonArrivalProcess(2, streams, 'f/main')
            service = GammaServiceTime(1, 1, streams, 'f')
            return [(arrivals.next_iat(), service.sample()) for _ in range(10)]

        first = sample()
        second = sample()
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
