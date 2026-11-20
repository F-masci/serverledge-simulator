"""Runtime delle funzioni, immagini e limiti di concorrenza."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from sim.utils.validation import finite_time, nonempty_string


_BUILTIN_IMAGES = {
    'python314': 'grussorusso/serverledge-python314',
    'python312ml': 'grussorusso/serverledge-python312ml',
    'go125': 'grussorusso/serverledge-go125',
    'nodejs17ng': 'grussorusso/serverledge-nodejs17ng',
}


@dataclass(frozen=True, kw_only=True)
class Runtime(ABC):
    """Politica di normalizzazione della concorrenza alla registrazione."""

    _runtimes: ClassVar[dict[tuple[str, str, float], 'Runtime']] = {}
    startup_time: float = 0.0

    @staticmethod
    def get_runtime(
        name: str, *, custom_image: str | None = None, startup_time: float = 0.0,
        startup_times: Mapping[str, float] | None = None,
    ) -> 'Runtime':
        """Restituisce l'istanza condivisa per runtime, immagine e tempo di avvio."""

        # Cerca il nome tra i runtime integrati e risolve l'immagine Docker.
        builtin = name in _BUILTIN_IMAGES
        if builtin:
            if custom_image is not None:
                raise ValueError('custom_image è ammesso solo per il runtime custom')
            image = _BUILTIN_IMAGES[name]
        else:
            # Un nome non integrato può indicare direttamente un'immagine custom.
            if name == 'custom' and custom_image is None:
                raise ValueError('Immagine custom obbligatoria')
            image = nonempty_string(custom_image if custom_image is not None else name, 'custom_image')

        # Usa il tempo configurato per l'immagine, se presente nello scenario.
        if startup_times is not None:
            startup_time = startup_times.get(image, startup_time)
        startup_time = finite_time(startup_time, 'runtime.startup_time')

        # Per i custom l'identità dipende dall'immagine, anche se dichiarata in forme diverse.
        key = (name if builtin else 'custom', image, startup_time)
        if key in Runtime._runtimes:
            return Runtime._runtimes[key]

        # Costruisce e registra il runtime solo alla prima richiesta.
        if builtin:
            runtime = BuiltinRuntime(name, startup_time=startup_time)
        else:
            runtime = CustomRuntime(image, startup_time=startup_time)
        return Runtime._runtimes.setdefault(key, runtime)

    def __post_init__(self) -> None:
        if finite_time(self.startup_time, 'runtime.startup_time') < 0:
            raise ValueError('runtime.startup_time deve essere non negativo')

    @abstractmethod
    def effective_concurrency(self, requested: int) -> int:
        """Restituisce il limite applicabile ai container della funzione."""


@dataclass(frozen=True)
class BuiltinRuntime(Runtime):
    name: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.name not in _BUILTIN_IMAGES:
            raise ValueError(f'Runtime sconosciuto: {self.name}')

    @property
    def image(self) -> str:
        return _BUILTIN_IMAGES[self.name]

    def effective_concurrency(self, requested: int) -> int:
        return 1 if self.name == 'nodejs17ng' else max(1, requested)


@dataclass(frozen=True)
class CustomRuntime(Runtime):
    image: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.image:
            raise ValueError('Immagine custom obbligatoria')

    def effective_concurrency(self, requested: int) -> int:
        return 1
