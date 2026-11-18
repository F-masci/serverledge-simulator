"""Selezione iniziale del ring destinatario di una richiesta."""

from collections.abc import Iterable

from .ring import ArchitectureRing
from .request import Request


class LoadBalancer:
    def submit(
        self, request: Request, rings: Iterable[ArchitectureRing],
    ) -> ArchitectureRing | None:
        """Restituisce il primo ring non vuoto compatibile, senza inoltrare.

        L'ordine è quello di registrazione nell'infrastruttura.
        """
        return next((ring for ring in rings if len(ring) and request.function.supports_arch(ring.arch)), None)
