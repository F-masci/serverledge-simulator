"""Ingresso geografico delle invocazioni e gestione della topologia regionale."""

from collections.abc import Mapping
from enum import StrEnum
import json
from types import MappingProxyType
from typing import TYPE_CHECKING

from .infrastructure import Infrastructure
from .load_balancer import LoadBalancer
from .node import Node
from .request import Request

if TYPE_CHECKING:
    from sim.bindings import NodeBindings
    from sim.io.scenario import Scenario
    from .ring import ArchitectureRing


class RegionMode(StrEnum):
    """Modalità di esecuzione di una regione di simulazione."""
    SIMULATED = 'simulated'              # Regione con infrastruttura di anelli, nodi e container simulati.
    ALWAYS_SUCCEEDS = 'always_succeeds'  # Regione sintetica che accetta sempre con tempo di servizio fisso.


class Region:
    """
    Rappresenta una regione geografica di ingresso o di offloading.

    In modalità SIMULATED costruisce e possiede l'infrastruttura locale
    (load balancer, anelli e nodi) a partire dallo scenario. In modalità
    ALWAYS_SUCCEEDS accetta direttamente ogni invocazione con un tempo di
    servizio fisso senza allocare container.
    """

    def __init__(
        self,
        name: str,
        *,
        scenario: 'Scenario | None' = None,
        node_bindings: 'NodeBindings | None' = None,
    ) -> None:
        # Identificativo univoco della regione
        self._name = name

        # Configurazione di esecuzione, latenze e infrastruttura locale
        self._service_time_s: float = 0.0
        self._offload_latency_s: float = 0.0
        self._offload_latency_by_source_s: Mapping[str, float] = MappingProxyType({})
        self._has_load_balancer: bool = True
        self._infrastructure: Infrastructure | None = None

        if scenario is not None:
            config = scenario.regions[name]
            self._service_time_s = config.service_time_s
            self._offload_latency_s = config.offload_latency_s
            self._offload_latency_by_source_s = MappingProxyType(
                dict(config.offload_latency_by_source_s)
            )
            self._has_load_balancer = config.has_load_balancer
            if config.infrastructure is not None:
                self._infrastructure = self._build_infrastructure(
                    scenario, node_bindings=node_bindings,
                )

    ### Identification ###

    @property
    def name(self) -> str:
        return self._name

    ### Region State ###

    @property
    def mode(self) -> RegionMode:
        """Restituisce la modalità di esecuzione della regione."""
        return RegionMode.SIMULATED if self._infrastructure is not None else RegionMode.ALWAYS_SUCCEEDS

    @property
    def is_simulated(self) -> bool:
        """Indica se la regione dispone di un'infrastruttura simulata."""
        return self.mode is RegionMode.SIMULATED

    @property
    def is_always_succeeds(self) -> bool:
        """Indica se la regione accetta sempre le invocazioni senza container."""
        return self.mode is RegionMode.ALWAYS_SUCCEEDS

    @property
    def service_time_s(self) -> float:
        return self._service_time_s

    @property
    def offload_latency_s(self) -> float:
        return self._offload_latency_s

    @property
    def offload_latency_by_source_s(self) -> Mapping[str, float]:
        return self._offload_latency_by_source_s

    @property
    def has_load_balancer(self) -> bool:
        return self._has_load_balancer

    ### Infrastructure ###

    @property
    def infrastructure(self) -> Infrastructure | None:
        return self._infrastructure

    def offload_latency_from(self, source: str) -> float:
        """Ritardo totale di andata e ritorno dalla regione sorgente."""
        return self._offload_latency_by_source_s.get(source, self._offload_latency_s)

    def first_registered_node(self) -> Node | None:
        """Restituisce il primo nodo registrato in ordine lessicografico machine_tag/nome."""
        infrastructure = self._infrastructure
        if infrastructure is None:
            return None

        # Come nel registry Go, ordina per chiave machine_tag/nome.
        return min(
            (node for ring in infrastructure.rings.values() for node in ring),
            key=lambda node: f'{node.id.machine_tag}/{node.id.key}',
            default=None,
        )

    #################################
    ### Internal state management ###
    #################################

    ## Infra management ##

    def _build_infrastructure(
        self,
        scenario: 'Scenario',
        *,
        node_bindings: 'NodeBindings | None' = None,
    ) -> Infrastructure:
        """Costruisce load balancer, anelli e nodi della regione a partire dallo scenario."""
        config = scenario.regions[self._name].infrastructure
        assert config is not None

        # Crea il load balancer con uno stream PRNG dedicato alla regione.
        stream_id = 'load_balancer' if self._name == scenario.primary_region else f'load_balancer/{self._name}'
        load_balancer = LoadBalancer.create(scenario.load_balancer, seed=scenario.seed, stream_id=stream_id)
        infrastructure = Infrastructure(
            load_balancer,
            region=self,
            node_bindings=node_bindings,
        )

        # Popola gli anelli e i rispettivi nodi con i parametri dello scenario.
        for ring in config.rings:
            self._create_ring(
                infrastructure, ring.id,
                arch=ring.arch, memory=ring.memory_mb, cpus=ring.cpus, replicas=ring.replicas,
            )
            for node in ring.nodes:
                self._create_node(
                    infrastructure, ring.id, node.name, node.machine_tag,
                    idle_timeout=scenario.containers.idle_timeout,
                    queue_capacity=scenario.scheduler.queue_capacity,
                )

        return infrastructure

    @staticmethod
    def _create_ring(
        infrastructure: Infrastructure,
        ring_id: str,
        *,
        arch: str,
        memory: int,
        cpus: float,
        replicas: int,
    ) -> ArchitectureRing:
        """Crea un anello sull'infrastruttura della regione (ereditando i binding configurati)."""
        return infrastructure.create_ring(
            ring_id, arch=arch, memory=memory, cpus=cpus, replicas=replicas,
        )

    @staticmethod
    def _create_node(
        infrastructure: Infrastructure,
        ring_id: str,
        name: str,
        machine_tag: str | None = None,
        *,
        idle_timeout: float = 600,
        queue_capacity: int = 0,
    ) -> Node:
        """Crea un nodo nell'anello indicato (applicando i binding al livello dell'anello)."""
        return infrastructure.create_node(
            ring_id, name, machine_tag,
            idle_timeout=idle_timeout,
            queue_capacity=queue_capacity,
        )

    ## Request management ##

    def submit(
        self, request: Request, *, now: float, node: Node | None = None,
    ) -> Node | None:
        """Registra l'ingresso della richiesta nella regione e ne gestisce l'ammissione."""
        request.enter_region(self, now)

        # In modalità sintetica la richiesta viene accettata direttamente sulla regione.
        infrastructure = self._infrastructure
        if infrastructure is None:
            request.accept_on_region(self)
            return None

        # In assenza di load balancer e di un nodo esplicito si indirizza al primo nodo registrato.
        if node is None and not self._has_load_balancer:
            node = self.first_registered_node()
            if node is None:
                request.reject('no_destination')
                return None

        return infrastructure.submit(request, now=now, node=node)

    #####################
    ### Serialization ###
    #####################

    def __str__(self) -> str:
        """Restituisce una descrizione leggibile della regione."""
        return f'Region(name={self._name}, mode={self.mode.value})'

    def to_dict(self) -> dict:
        """Restituisce la configurazione e lo stato della regione in formato serializzabile."""
        return {
            'name': self._name,
            'mode': self.mode.value,
            'service_time_s': self._service_time_s,
            'offload_latency_s': self._offload_latency_s,
            'offload_latency_by_source_s': dict(self._offload_latency_by_source_s),
            'has_load_balancer': self._has_load_balancer,
            'infrastructure': self._infrastructure.to_dict() if self._infrastructure is not None else None,
        }

    def to_json(self) -> str:
        """Serializza la regione per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<Region name={self._name} mode={self.mode.value}>'
