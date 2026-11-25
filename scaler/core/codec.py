"""Codifica e decodifica dello spazio discreto (1 + 2R) e della maschera di ammissibilità."""

import numpy as np

from .. import RingObservation
from ..contracts import ScalingActionKind, ZoneObservation


class ActionCodec:
    """Traduce tra indice discreto 0..2R e coppia (ScalingActionKind, ring_id).

    Esempio con 2 ring (`ring_ids = ('edge-x86', 'edge-arm')`):
        - `_ring_ids`: `('edge-x86', 'edge-arm')`
        - `_action_space` (indice -> azione):
            - `0` -> `(ScalingActionKind.HOLD, None)`
            - `1` -> `(ScalingActionKind.SCALE_OUT, 'edge-x86')`
            - `2` -> `(ScalingActionKind.SCALE_IN,  'edge-x86')`
            - `3` -> `(ScalingActionKind.SCALE_OUT, 'edge-arm')`
            - `4` -> `(ScalingActionKind.SCALE_IN,  'edge-arm')`
        - `_index` (azione -> indice): mappa inversa `{(HOLD, None): 0, (SCALE_OUT, 'edge-x86'): 1, ...}`
        - `decode(3)` -> `(ScalingActionKind.SCALE_OUT, 'edge-arm')`
        - `encode(ScalingActionKind.SCALE_IN, 'edge-x86')` -> `2`
    """

    def __init__(self, ring_ids: tuple[str, ...]) -> None:
        # Elenco ordinato e stabile degli ID dei ring della regione, es.: ('edge-x86', 'edge-arm').
        # L'ordine fissa la corrispondenza tra ogni ring e gli indici discreti dell'agente RL.
        self._ring_ids = tuple(ring_ids)

        # Costruisce la tabella indice -> (tipo_azione, ring_id):
        #   0 -> (HOLD, None)
        #   1 + 2*r -> (SCALE_OUT, ring_r)
        #   2 + 2*r -> (SCALE_IN,  ring_r)
        actions: list[tuple[ScalingActionKind, str | None]] = [(ScalingActionKind.HOLD, None)]
        for ring_id in self._ring_ids:
            actions.append((ScalingActionKind.SCALE_OUT, ring_id))
            actions.append((ScalingActionKind.SCALE_IN, ring_id))

        # Tupla immutabile di lunghezza 1 + 2R usata da decode(action_index).
        self._action_space: tuple[tuple[ScalingActionKind, str | None], ...] = tuple(actions)

        # Mappa inversa (ScalingActionKind, ring_id | None) -> indice discreto 0..2R,
        # usata da encode(kind, ring_id) per lookup O(1).
        self._index = {pair: idx for idx, pair in enumerate(self._action_space)}

    @property
    def ring_ids(self) -> tuple[str, ...]:
        return self._ring_ids

    @property
    def action_dim(self) -> int:
        return len(self._action_space)

    def decode(self, action_index: int) -> tuple[ScalingActionKind, str | None]:
        """Decodifica un indice discreto in (ScalingActionKind, ring_id | None)."""
        return self._action_space[action_index]

    def encode(self, kind: ScalingActionKind, ring_id: str | None = None) -> int:
        """Codifica una coppia (ScalingActionKind, ring_id) nel suo indice discreto."""
        return self._index[(kind, ring_id)]

    def action_mask(self, observation: ZoneObservation) -> np.ndarray:
        """Costruisce la maschera booleana delle azioni ammissibili dall'osservazione di zona."""

        # Crea un dizionario per mappare gli ID dei ring ai loro oggetti RingObservation.
        by_id: dict[str, RingObservation] = {ring.ring_id: ring for ring in observation.rings}

        # Crea una maschera booleana per le azioni ammissibili.
        # Inizia con tutte le azioni disabilitate.
        mask = np.zeros(len(self._action_space), dtype=bool)
        mask[0] = True  # HOLD resta sempre disponibile.

        # Verifica se ogni azione è ammissibile in base alle condizioni del ring.
        for idx, (kind, ring_id) in enumerate(self._action_space[1:], start=1):
            assert ring_id is not None

            ring: RingObservation | None = by_id.get(ring_id)
            if ring is None:
                continue

            # Se il ring è disponibile, verifica se l'azione è ammissibile.
            if kind is ScalingActionKind.SCALE_OUT:
                mask[idx] = ring.can_scale_out
            elif kind is ScalingActionKind.SCALE_IN:
                mask[idx] = ring.can_scale_in

        return mask
