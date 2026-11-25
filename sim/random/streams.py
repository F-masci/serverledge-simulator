"""Creazione centralizzata di flussi pseudocasuali identificati per scopo."""

from enum import IntEnum
from hashlib import sha256
import json
import platform
import struct

import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence

from sim.utils.validation import nonnegative_integer


class StreamKind(IntEnum):
    """Codici persistenti degli scopi PRNG: non rinumerare quando si aggiungono nuovi scopi."""
    ARRIVAL = 1   # Generazione degli interarrivi delle invocazioni.
    SERVICE = 2   # Campionamento delle durate di servizio.
    ROUTING = 3   # Scelte casuali di instradamento e load balancing.
    EVICTION = 4  # Ordine casuale dei pool durante l'eviction per pressione di memoria.
    SCALING = 5   # Scelte stocastiche dello scaler RL ed eventi di provisioning.


class RandomStreams:
    """
    Registro dei generatori pseudocasuali di una singola replica.

    Una chiave (scopo, identificativo) restituisce sempre lo stesso generatore
    in avanzamento. Un nuovo registro con gli stessi parametri riparte dall'inizio.
    Gli identificativi devono essere stabili e non dipendere da ordine o worker.
    """

    SCHEMA_VERSION = 1

    def __init__(self, seed: int, replication: int = 0) -> None:
        # Parametri di riproducibilità della replica.
        self._seed = nonnegative_integer(seed, 'seed')
        self._replication = nonnegative_integer(replication, 'replication')

        # Cache dei generatori attivi indicizzati per (scopo, identificativo).
        self._streams: dict[tuple[StreamKind, str], Generator] = {}

    ### Stream State ###

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def replication(self) -> int:
        return self._replication

    #################################
    ### Internal state management ###
    #################################

    ## Stream registry ##

    def stream(self, kind: StreamKind, stream_id: str) -> Generator:
        """Restituisce il generatore associato alla coppia (kind, stream_id), creandolo se assente."""
        if not isinstance(kind, StreamKind):
            raise ValueError('Scopo del flusso non riconosciuto')
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError('Identificativo del flusso non vuoto obbligatorio')

        key = (kind, stream_id)
        if key not in self._streams:
            # SHA-256 definisce una codifica stabile e indipendente dal processo (hash() varia fra run).
            words = struct.unpack('<8I', sha256(stream_id.encode('utf-8')).digest())
            sequence = SeedSequence([
                self.SCHEMA_VERSION, self._replication, int(kind), *words, self._seed,
            ])
            self._streams[key] = Generator(PCG64DXSM(sequence))

        return self._streams[key]

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile del registro di flussi."""
        return f'RandomStreams(seed={self._seed}, replication={self._replication}, streams={len(self._streams)})'

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

    def to_dict(self) -> dict:
        """Restituisce i metadati del registro in formato serializzabile."""
        return self.metadata()

    def to_json(self) -> str:
        """Serializza i metadati del registro per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<RandomStreams seed={self._seed} replication={self._replication} streams={len(self._streams)}>'
