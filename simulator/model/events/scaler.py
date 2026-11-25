"""Eventi discreti del ciclo di controllo e delle transizioni dello scaler regionale."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .event import Event

if TYPE_CHECKING:
    from simulator.model.topology.region import Region


@dataclass(frozen=True)
class ScalerStepEvent(Event):
    """Tick periodico (intervallo Delta) del ciclo di controllo RL di una regione."""

    region: 'Region'

    def execute(self) -> None:
        """Invoca il passo decisionale dello scaler della regione."""
        scaler = self.region.scaler
        if scaler is not None:
            scaler.step(self.ctx)


@dataclass(frozen=True)
class NodeProvisionedEvent(Event):
    """Completamento asincrono dell'acquisizione e boot di un nodo su un ring."""

    region: 'Region'
    ring_id: str
    op_id: str

    def execute(self) -> None:
        """Notifica allo scaler regionale la fine del boot del nodo."""
        scaler = self.region.scaler
        if scaler is not None:
            scaler.on_node_provisioned(self.ring_id, self.op_id, self.ctx.now)


@dataclass(frozen=True)
class NodeDrainedEvent(Event):
    """Completamento del drain e dello spegnimento di un nodo consolidato."""

    region: 'Region'
    ring_id: str
    op_id: str

    def execute(self) -> None:
        """Notifica allo scaler regionale il rilascio definitivo del nodo drenato."""
        scaler = self.region.scaler
        if scaler is not None:
            scaler.on_node_drained(self.ring_id, self.op_id, self.ctx.now)
