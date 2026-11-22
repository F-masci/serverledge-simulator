"""Creazione centralizzata di flussi pseudocasuali identificati per scopo."""

import platform
import struct
from enum import IntEnum
from hashlib import sha256

import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence

from sim.utils.validation import nonnegative_integer


class StreamKind(IntEnum):
    """Codici persistenti: non rinumerare quando si aggiungono nuovi scopi."""

    ARRIVAL = 1
    SERVICE = 2
    ROUTING = 3


class RandomStreams:
    """Registro dei generatori di una singola replica.

    Una chiave (scopo, identificativo) restituisce sempre lo stesso generatore
    in avanzamento. Un nuovo registro con stessi parametri riparte dall'inizio.
    Gli identificativi devono essere stabili e non dipendere da ordine o worker.
    """

    SCHEMA_VERSION = 1

    def __init__(self, seed: int, replication: int = 0) -> None:
        self._seed = nonnegative_integer(seed, 'seed')
        self._replication = nonnegative_integer(replication, 'replication')
        self._streams: dict[tuple[StreamKind, str], Generator] = {}

    def stream(self, kind: StreamKind, stream_id: str) -> Generator:
        if not isinstance(kind, StreamKind):
            raise ValueError('Scopo del flusso non riconosciuto')
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError('Identificativo del flusso non vuoto obbligatorio')
        key = (kind, stream_id)
        if key not in self._streams:
            # SHA-256 definisce una codifica stabile; hash() varia fra processi.
            words = struct.unpack('<8I', sha256(stream_id.encode('utf-8')).digest())
            sequence = SeedSequence([
                self.SCHEMA_VERSION, self._replication, int(kind), *words, self._seed,
            ])
            self._streams[key] = Generator(PCG64DXSM(sequence))
        return self._streams[key]

    def metadata(self) -> dict:
        """Descrive inizializzazione e ambiente; non è un checkpoint dello stato."""
        return {
            'seed': self._seed,
            'replication': self._replication,
            'bit_generator': 'PCG64DXSM',
            'seed_sequence': 'SeedSequence',
            'schema_version': self.SCHEMA_VERSION,
            'id_encoding': 'SHA-256 UTF-8, eight little-endian uint32',
            'numpy_version': np.__version__,
            'python_version': platform.python_version(),
            'streams': [
                {'kind': kind.name, 'id': stream_id}
                for kind, stream_id in sorted(self._streams)
            ],
        }
