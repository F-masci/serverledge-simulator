"""Specifica dei binding di ciclo di vita da associare all'attuatore regionale."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulator.model.topology.node import Node
    from simulator.model.topology.region import Region
    from simulator.model.scaling.actuator import DrainingOp, ProvisioningOp, RegionActuator


@dataclass(frozen=True)
class ActuatorBindings:
    """
    Contenitore immutabile dei riferimenti alle callback di un `RegionActuator`.

    Permette a `Simulation` di gestire lo scheduling degli eventi discreti di
    provisioning e drain e la creazione dei nuovi nodi secondo lo scenario,
    mantenendo `RegionActuator` disaccoppiato dal motore e dal contesto di simulazione.
    """

    ### Provisioning Callbacks ###

    on_provision_started: Callable[[Region, str, ProvisioningOp, float], None] | None = None
    on_provision_completed: Callable[[Region, str, str], Node] | None = None

    ### Drain Callbacks ###

    on_drain_ready: Callable[[Region, str, DrainingOp, float], None] | None = None

    ###############
    ### Binding ###
    ###############

    def bind(self, actuator: RegionActuator) -> None:
        """Associa all'attuatore tutte le callback di ciclo di vita configurate."""
        if self.on_provision_started is not None:
            actuator.on_provision_started = self.on_provision_started
        if self.on_provision_completed is not None:
            actuator.on_provision_completed = self.on_provision_completed
        if self.on_drain_ready is not None:
            actuator.on_drain_ready = self.on_drain_ready
