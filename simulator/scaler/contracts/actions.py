"""Comandi di scaling ed esiti di attuazione (mappabili 1-a-1 su Protobuf/gRPC)."""

from dataclasses import dataclass
from enum import StrEnum


class ScalingActionKind(StrEnum):
    """Tipo di operazione di scaling decidibile dallo Zone Controller."""
    HOLD = 'hold'            # Mantieni: non avvia nuove operazioni; quelle in corso proseguono.
    SCALE_OUT = 'scale_out'  # Avvia: acquisisce e registra un nuovo nodo nel pool/ring.
    SCALE_IN = 'scale_in'    # Consolida: seleziona un nodo, lo esclude dal routing e ne attende il drain.


class CommandOutcome(StrEnum):
    """Esito della richiesta di scaling gestita dall'Infra Controller."""
    ACCEPTED = 'accepted'
    REJECTED = 'rejected'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    FAILED = 'failed'


@dataclass(frozen=True)
class ScalingCommand:
    """Comando di scaling emesso dallo Zone Controller verso l'Infra Controller."""

    command_id: str
    decision_id: int
    ring_id: str | None
    action: ScalingActionKind
    state_version: int
    issued_at_s: float


@dataclass(frozen=True)
class ScalingResult:
    """Ricevuta o esito dell'attuazione di un comando di scaling."""

    command: ScalingCommand
    outcome: CommandOutcome
    reason: str | None = None
    node_name: str | None = None
    completed_at_s: float | None = None
