"""Verifiche di riproducibilità, separazione e distribuzioni dei flussi.

Riferimenti:
- NumPy, Parallel random number generation: SeedSequence e combinazione di ID
  deterministici con seed radice; https://numpy.org/doc/stable/reference/random/parallel.html
- NumPy, PCG64DXSM: algoritmo esplicito e compatibilità del flusso di interi;
  https://numpy.org/doc/stable/reference/random/bit_generators/pcg64dxsm.html
- NumPy, Generator.exponential e Generator.gamma: parametrizzazione dei campioni.
- NumPy, Compatibility policy: dipendenza da ambiente e sequenza di chiamate;
  https://numpy.org/doc/stable/reference/random/compatibility.html

I test di separazione verificano il contratto del registro, non dimostrano
indipendenza statistica dei PRNG. Le verifiche dei momenti sono controlli
su campioni a seed fisso, non certificazioni delle distribuzioni di NumPy.
"""

import json
import os
import subprocess
import sys
import unittest

import numpy as np
from simulator.random import RandomStreams, StreamKind, PoissonArrivalProcess, GammaServiceTime


class RandomStreamsTest(unittest.TestCase):
    def test_repeated_lookup_continues_the_same_stream(self):
        """La stessa chiave restituisce un generatore in avanzamento, senza reseed."""
        streams = RandomStreams(1234)
        first = streams.stream(StreamKind.ARRIVAL, 'f/main')
        a = first.random(4)
        again = streams.stream(StreamKind.ARRIVAL, 'f/main')
        self.assertIs(first, again)
        b = again.random(4)
        reference = RandomStreams(1234).stream(StreamKind.ARRIVAL, 'f/main')
        np.testing.assert_array_equal(np.concatenate((a, b)), reference.random(8))

    def test_creation_order_and_extra_draws_do_not_change_other_streams(self):
        """Creare altri flussi o consumare arrivi non cambia il servizio di f."""
        baseline = RandomStreams(42)
        expected = baseline.stream(StreamKind.SERVICE, 'f').random(20)
        changed = RandomStreams(42)
        changed.stream(StreamKind.ARRIVAL, 'f').random(1000)
        changed.stream(StreamKind.SERVICE, 'unrelated').random(30)
        np.testing.assert_array_equal(expected, changed.stream(StreamKind.SERVICE, 'f').random(20))

    def test_purpose_identity_and_replication_select_distinct_sequences(self):
        """Scopo, ID e replica sono dimensioni separate della chiave di inizializzazione."""
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
        """Processi con hash Python differenti ricostruiscono lo stesso flusso."""
        code = ('import json; from simulator.random import RandomStreams, StreamKind; '
                'print(json.dumps(RandomStreams(42).stream(StreamKind.SERVICE, "f").random(5).tolist()))')
        outputs = []
        for hash_seed in ('1', '999'):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            outputs.append(subprocess.check_output([sys.executable, '-c', code], env=env, text=True))
        self.assertEqual(outputs[0], outputs[1])

    def test_metadata_records_initialization_without_consuming_samples(self):
        """Il manifest è JSON e non consuma estrazioni; non è un checkpoint."""
        streams = RandomStreams(42, 3)
        rng = streams.stream(StreamKind.SERVICE, 'f')
        metadata = streams.metadata()
        json.dumps(metadata)
        self.assertEqual(metadata['replication'], 3)
        self.assertEqual(metadata['bit_generator'], 'PCG64DXSM')
        self.assertEqual(metadata['numpy_version'], np.__version__)
        self.assertEqual(metadata['streams'], [{'kind': 'SERVICE', 'id': 'f'}])
        np.testing.assert_array_equal(rng.random(5), RandomStreams(42, 3).stream(StreamKind.SERVICE, 'f').random(5))

    def test_invalid_initialization_is_rejected(self):
        """Seed/replica interi non negativi e chiavi esplicite sono obbligatori."""
        for value in (-1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    RandomStreams(value)
                with self.assertRaises(ValueError):
                    RandomStreams(1, value)
        for kind, name in ((1, 'f'), (StreamKind.ARRIVAL, '')):
            with self.assertRaises(ValueError):
                RandomStreams(1).stream(kind, name)


class DistributionsTest(unittest.TestCase):
    def test_poisson_interarrivals_have_expected_mean(self):
        """Rate 4/s produce intervalli non negativi con media circa 0.25 s."""
        process = PoissonArrivalProcess(4, RandomStreams(10), 'f/main')
        samples = np.array([process.next_iat() for _ in range(30000)])
        self.assertTrue(np.all(samples >= 0))
        self.assertAlmostEqual(samples.mean(), 0.25, delta=0.01)

    def test_gamma_service_has_expected_mean_and_scv(self):
        """La Gamma usa media e SCV richiesti, senza cold start o attesa."""
        service = GammaServiceTime(2, 0.5, RandomStreams(10), 'f')
        samples = np.array([service.sample() for _ in range(30000)])
        self.assertTrue(np.all(samples >= 0))
        self.assertAlmostEqual(samples.mean(), 2, delta=0.05)
        self.assertAlmostEqual(samples.var() / samples.mean() ** 2, 0.5, delta=0.04)

    def test_distribution_samples_reproduce_after_reconstruction(self):
        """Ricostruire la replica riproduce anche le trasformazioni, non solo gli interi."""
        def sample():
            streams = RandomStreams(99, 2)
            arrivals = PoissonArrivalProcess(2, streams, 'f/main')
            service = GammaServiceTime(1, 1, streams, 'f')
            return [(arrivals.next_iat(), service.sample()) for _ in range(10)]
        self.assertEqual(sample(), sample())

    def test_invalid_distribution_parameters_are_rejected(self):
        """Niente distribuzioni con tassi, medie o SCV nulli, negativi o non finiti."""
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    PoissonArrivalProcess(value, RandomStreams(1), 'f')
                with self.assertRaises(ValueError):
                    GammaServiceTime(value, 1, RandomStreams(1), 'f')
                with self.assertRaises(ValueError):
                    GammaServiceTime(1, value, RandomStreams(1), 'f')


if __name__ == '__main__':
    unittest.main()
