"""Ingresso geografico delle invocazioni e gestione della topologia regionale."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
import json
from types import MappingProxyType
from typing import TYPE_CHECKING

from sim.random.streams import RandomStreams, StreamKind
from .infrastructure import Infrastructure
from .load_balancer import LoadBalancer
from .node import Node
from .request import Request
from .scaling import Scaler

if TYPE_CHECKING:
    from numpy.random import Generator
    from ray.rllib.core.rl_module.rl_module import RLModule
    from sim.bindings import ActuatorBindings, NodeBindings
    from sim.io.scenario import Scenario
    from sim.scaler.rllib import RLlibPPOPolicy
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
        actuator_bindings: 'ActuatorBindings | None' = None,
        scaler_agent: 'RLlibPPOPolicy | RLModule | None' = None,
    ) -> None:
        # Identificativo univoco della regione
        self._name = name

        # Configurazione di esecuzione, latenze, infrastruttura e scaler locale
        self._service_time_s: float = 0.0
        self._offload_latency_s: float = 0.0
        self._offload_latency_by_source_s: Mapping[str, float] = MappingProxyType({})
        self._has_load_balancer: bool = True
        self._infrastructure: Infrastructure | None = None
        self._scaler: 'Scaler | None' = None

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
                self._scaler = self._build_scaler(scenario, actuator_bindings=actuator_bindings, agent=scaler_agent)

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

    @property
    def scaler(self) -> 'Scaler | None':
        """Scaler RL della regione, o None se l'autoscaling è disabilitato."""
        return self._scaler

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
        region_config = scenario.regions[self._name]
        config = region_config.infrastructure
        assert config is not None

        # Crea il load balancer con uno stream PRNG dedicato alla regione.
        # La modalità e il numero di repliche possono essere definiti sulla singola regione
        # o ereditati dallo scenario (default 128; se non specificati, ArchitectureUnaware
        # eredita le repliche dei soli anelli non vuoti se uniformi).
        streams = RandomStreams(scenario.seed)
        stream_id = 'load_balancer' if self._name == scenario.primary_region else f'load_balancer/{self._name}'
        lb_mode = region_config.load_balancer or scenario.load_balancer
        explicit_replicas = (
            region_config.load_balancer_replicas
            if region_config.load_balancer_replicas is not None
            else scenario.load_balancer_replicas
        )
        if explicit_replicas is not None:
            lb_replicas = explicit_replicas
        else:
            active_replicas = {ring.replicas or 128 for ring in config.rings if ring.nodes}
            lb_replicas = next(iter(active_replicas)) if len(active_replicas) == 1 else 128
        load_balancer = LoadBalancer.create(
            lb_mode, seed=scenario.seed, stream_id=stream_id, replicas=lb_replicas,
        )
        infrastructure = Infrastructure(
            load_balancer,
            region=self,
            node_bindings=node_bindings,
        )

        # Popola gli anelli e i rispettivi nodi con i parametri dello scenario.
        # Un ring senza replicas esplicite eredita load_balancer_replicas (come lb.replicas in Go).
        default_ring_replicas = explicit_replicas if explicit_replicas is not None else 128
        for ring in config.rings:
            self._create_ring(
                infrastructure, ring.id,
                arch=ring.arch, memory=ring.memory_mb, cpus=ring.cpus,
                replicas=ring.replicas if ring.replicas is not None else default_ring_replicas,
                machine_tag=ring.machine_tag,
            )
            for node in ring.nodes:
                eviction_rng = streams.stream(StreamKind.EVICTION, f'{self._name}/{ring.id}/{node.name}')
                self._create_node(
                    infrastructure, ring.id, node.name, node.machine_tag,
                    idle_timeout=scenario.containers.idle_timeout,
                    queue_capacity=scenario.scheduler.queue_capacity,
                    eviction_rng=eviction_rng,
                )

        return infrastructure

    def _build_scaler(
        self,
        scenario: 'Scenario',
        *,
        actuator_bindings: 'ActuatorBindings | None' = None,
        agent: 'RLlibPPOPolicy | RLModule | None' = None,
    ) -> 'Scaler | None':
        """Costruisce lo scaler regionale se configurato sulla regione o sullo scenario."""
        region_config = scenario.regions[self._name]
        scaler_config = region_config.scaler if region_config.scaler is not None else scenario.scaler

        # Se lo scaler non è abilitato o la regione non ha infrastruttura, restituisce None.
        if not scaler_config.is_enabled:
            return None

        # Se la regione non ha infrastruttura o anelli, lo scaler non può essere creato.
        if self._infrastructure is None or not self._infrastructure.rings:
            return None

        config = region_config.infrastructure
        assert config is not None

        # Costruisce la mappa dei parametri di scaling per ogni anello.
        ring_configs = {
            ring.id: ring.scaling
            for ring in config.rings
            if ring.scaling is not None
        }

        # Crea lo scaler con uno stream PRNG dedicato alla regione.
        stream_id = 'scaler' if self._name == scenario.primary_region else f'scaler/{self._name}'
        return Scaler.create(
            scaler_config,
            region=self,
            ring_configs=ring_configs,
            actuator_bindings=actuator_bindings,
            seed=scenario.seed,
            stream_id=stream_id,
            agent=agent,
        )

    @staticmethod
    def _create_ring(
        infrastructure: Infrastructure,
        ring_id: str,
        *,
        arch: str,
        memory: int,
        cpus: float,
        replicas: int,
        machine_tag: str | None = None,
    ) -> ArchitectureRing:
        """Crea un anello sull'infrastruttura della regione (ereditando i binding configurati)."""
        return infrastructure.create_ring(
            ring_id, arch=arch, memory=memory, cpus=cpus, replicas=replicas, machine_tag=machine_tag,
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
        eviction_rng: 'Generator | None' = None,
    ) -> Node:
        """Crea un nodo nell'anello indicato (applicando i binding al livello dell'anello)."""
        return infrastructure.create_node(
            ring_id, name, machine_tag,
            idle_timeout=idle_timeout,
            queue_capacity=queue_capacity,
            eviction_rng=eviction_rng,
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

        # In assenza di load balancer e di un nodo esplicito si indirizza al primo nodo registrato,
        # lasciando allo scheduler del nodo le verifiche di architettura ed eventuale offloading.
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
            'scaler': self._scaler.to_dict() if self._scaler is not None else None,
        }

    def to_json(self) -> str:
        """Serializza la regione per la stampa o l'esportazione."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False)

    def __repr__(self) -> str:
        return f'<Region name={self._name} mode={self.mode.value}>'
