"""Normalizzazione delle osservazioni per ring e gestione dello storico breve."""

from collections import deque

import numpy as np

from ..config import RewardConfig
from ..contracts import FEATURES_PER_RING, ZoneObservation


class ObservationEncoder:
    """Trasforma una ZoneObservation nel vettore normalizzato (con storico breve) per l'agente RL.

    Per ciascun ring (nell'ordine fisso `ring_ids`) produce 13 feature numeriche:
        [0]  log(1 + recent_arrivals)
        [1]  log(1 + backlog_count)
        [2]  min(backlog_max_age_s / max(timeout_penalty_s, interval_s), 5.0)
        [3]  cpu_utilization                                         (in [0, 1])
        [4]  memory_utilization                                      (in [0, 1])
        [5]  ready_nodes / max_nodes                                 (in [0, 1])
        [6]  provisioning_nodes / max_nodes                          (in [0, 1])
        [7]  draining_nodes / max_nodes                              (in [0, 1])
        [8]  min(est_remaining_boot_s / max(boot_time_s, interval_s), 2.0)
        [9]  remaining_quota / max_nodes                             (in [0, 1])
        [10] min(interval_energy_kwh / k_ref, 10.0)
        [11] min(interval_delay_req_s / d_ref, 10.0)
        [12] log(1 + interval_rejections)

    Esempio con 2 ring (`('edge-x86', 'edge-arm')`) e `history_length = 2`:
        - singolo istante: 2 * 13 = 26 float (`[13 feature edge-x86, 13 feature edge-arm]`)
        - vettore finale concatenato sullo storico: 26 * 2 = 52 float
          (`[istante t-1 (26), istante t (26)]`), passato come `obs['observations']` alla rete RL.
    """

    def __init__(
        self,
        ring_ids: tuple[str, ...],
        reward_config: RewardConfig,
        *,
        interval_s: float,
        history_length: int = 1,
    ) -> None:
        # Ordine fisso dei ring: garantisce posizioni stabili delle feature tra training e inferenza.
        self._ring_ids = tuple(ring_ids)
        self._reward_config = reward_config
        self._interval_s = interval_s
        self._history_length = history_length

        # Dimensione del vettore di un singolo istante.
        self._single_dim = len(self._ring_ids) * FEATURES_PER_RING

        # Finestra scorrevole degli ultimi `history_length` istanti (inizializzata a zeri).
        self._history: deque[np.ndarray] = deque(
            (np.zeros(self._single_dim, dtype=np.float64) for _ in range(history_length)),
            maxlen=history_length,
        )

    @property
    def obs_dim(self) -> int:
        """Dimensione totale del vettore osservazione."""
        return self._single_dim * self._history_length

    def reset(self) -> None:
        """Azzera lo storico delle osservazioni."""
        self._history.clear()

        # Riempie la finestra con zeri.
        for _ in range(self._history_length):
            self._history.append(np.zeros(self._single_dim, dtype=np.float64))

    def encode_single(self, observation: ZoneObservation) -> np.ndarray:
        """Codifica e normalizza le 13 feature per ciascun ring senza mutare lo storico."""
        reward_config = self._reward_config

        # Crea un dizionario per mappare gli ID dei ring ai loro oggetti RingObservation.
        by_id = {ring.ring_id: ring for ring in observation.rings}

        # Calcola la scala per l'età del backlog.
        #
        # Questa scala è usata per normalizzare l'età del
        # backlog in modo che sia comparabile tra diversi ring.
        age_scale = max(reward_config.timeout_penalty_s, self._interval_s, 1e-6)

        # Costruisce il vettore delle feature concatenando le 13 feature di ciascun ring.
        features: list[float] = []
        for ring_id in self._ring_ids:
            ring = by_id[ring_id]
            max_nodes = max(1, ring.max_nodes)
            boot_scale = max(ring.boot_time_s, self._interval_s, 1e-6)
            features.extend((
                float(np.log1p(ring.recent_arrivals)),                    # [0] arrivi recenti (compresso log)
                float(np.log1p(ring.backlog_count)),                      # [1] dimensione backlog (compresso log)
                float(min(ring.backlog_max_age_s / age_scale, 5.0)),      # [2] età max backlog scalata e clippata
                float(ring.cpu_utilization),                              # [3] utilizzo CPU in [0, 1]
                float(ring.memory_utilization),                           # [4] utilizzo RAM busy in [0, 1]
                float(ring.ready_nodes / max_nodes),                      # [5] frazione nodi pronti
                float(ring.provisioning_nodes / max_nodes),               # [6] frazione nodi in boot/provisioning
                float(ring.draining_nodes / max_nodes),                   # [7] frazione nodi in drain
                float(min(ring.est_remaining_boot_s / boot_scale, 2.0)),  # [8] tempo residuo boot stimato
                float(ring.remaining_quota / max_nodes),                  # [9] frazione quota residua acquisibile
                float(min(ring.interval_energy_kwh / reward_config.k_ref, 10.0)),  # [10] energia intervallo scalata su K_ref
                float(min(ring.interval_delay_req_s / reward_config.d_ref, 10.0)), # [11] delay intervallo scalato su D_ref
                float(np.log1p(ring.interval_rejections)),                # [12] rifiuti intervallo (compresso log)
            ))

        return np.array(features, dtype=np.float64)

    def peek(self, observation: ZoneObservation) -> np.ndarray:
        """Restituisce `[storico precedente..., istante corrente]` senza avanzare la finestra."""
        raw = self.encode_single(observation)
        if self._history_length == 1:
            return raw

        # Scarta l'elemento più vecchio (indice 0, come farebbe append() con maxlen) e accoda `raw`.
        return np.concatenate(tuple(self._history)[1:] + (raw,))

    def push(self, observation: ZoneObservation) -> np.ndarray:
        """Accoda l'istante corrente nella finestra scorrevole e restituisce il vettore concatenato."""
        raw = self.encode_single(observation)
        self._history.append(raw)
        return np.concatenate(tuple(self._history))
