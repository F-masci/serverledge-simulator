"""Definizione delle funzioni e politiche di concorrenza dei runtime."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math


class Runtime(ABC):
    """Politica di normalizzazione della concorrenza alla registrazione."""

    @abstractmethod
    def effective_concurrency(self, requested: int) -> int:
        """Restituisce il limite applicabile ai container della funzione."""


@dataclass(frozen=True)
class BuiltinRuntime(Runtime):
    name: str

    def __post_init__(self) -> None:
        if self.name not in {'python314', 'python312ml', 'go125', 'nodejs17ng'}:
            raise ValueError(f'Runtime sconosciuto: {self.name}')

    def effective_concurrency(self, requested: int) -> int:
        return 1 if self.name == 'nodejs17ng' else max(1, requested)


@dataclass(frozen=True)
class CustomRuntime(Runtime):
    image: str

    def __post_init__(self) -> None:
        if not self.image:
            raise ValueError('Immagine custom obbligatoria')

    def effective_concurrency(self, requested: int) -> int:
        return 1


@dataclass(frozen=True)
class Function:
    """Configurazione registrata; i tempi di esecuzione non sono attributi reali.

    supported_archs contiene la compatibilità già risolta alla registrazione;
    per immagini custom deve essere fornita dal chiamante.
    """

    name: str
    memory_mb: int
    cpu_demand: float
    runtime: Runtime
    max_concurrency: int = 1
    supported_archs: tuple[str, ...] | None = None
    tag_pattern: str = ''
    handler: str = ''
    tar_function_code: str = ''

    def __post_init__(self) -> None:
        if not self.name or self.memory_mb < 1:
            raise ValueError('Nome e memoria positiva obbligatori')
        if not math.isfinite(self.cpu_demand) or self.cpu_demand <= 0:
            raise ValueError('CPU demand deve essere finita e positiva')
        if self.supported_archs is None and isinstance(self.runtime, CustomRuntime):
            raise ValueError('Specificare le architetture risolte per la immagine custom')
        archs = ('amd64', 'arm64') if self.supported_archs is None else self.supported_archs
        object.__setattr__(self, 'supported_archs', tuple(archs))
        object.__setattr__(self, 'max_concurrency',
                           self.runtime.effective_concurrency(self.max_concurrency))

    def supports_arch(self, arch: str) -> bool:
        return arch in self.supported_archs

    def __str__(self) -> str:
        return self.name
