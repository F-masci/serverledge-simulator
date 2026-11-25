"""Configurazione dei container idle e dei tempi di avvio per immagine.

Esempio TOML nello scenario (tutti i parametri sono opzionali):

    [containers]
    idle_timeout = 600.0      # opzionale (default: 600.0 s)
    cleanup_interval = 30.0   # opzionale (default: 30.0 s)

    [containers.startup_times]
    "grussorusso/serverledge-python314" = 0.15  # opzionale: immagine Docker -> secondi di avvio
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from sim.utils.validation import nonempty_string, nonnegative_number, positive_number


@dataclass(frozen=True)
class _ContainersConfig:
    """Politica per i container idle e tempi di avvio per immagine Docker."""

    idle_timeout: float
    cleanup_interval: float
    startup_times: Mapping[str, float]

    @classmethod
    def parse(cls, containers: object) -> '_ContainersConfig':
        """Parse della configurazione dei container."""
        if containers is None:
            containers = {}

        if not isinstance(containers, Mapping) or containers.keys() - {'idle_timeout', 'cleanup_interval', 'startup_times'}:
            raise ValueError('containers accetta solo idle_timeout, cleanup_interval e startup_times')
        idle_timeout = nonnegative_number(containers.get('idle_timeout', 600.0), 'containers.idle_timeout')
        cleanup_interval = positive_number(containers.get('cleanup_interval', 30.0), 'containers.cleanup_interval')

        startup_times = containers.get('startup_times', {})
        if not isinstance(startup_times, Mapping):
            raise ValueError('containers.startup_times deve associare immagini a durate')
        image_times = {}
        for image, duration in startup_times.items():
            image = nonempty_string(image, 'containers.startup_times (immagine)')
            duration = nonnegative_number(duration, f'containers.startup_times.{image}')
            image_times[image] = duration

        return cls(idle_timeout, cleanup_interval, MappingProxyType(image_times))

    def __getstate__(self) -> dict:
        return {
            'idle_timeout': self.idle_timeout,
            'cleanup_interval': self.cleanup_interval,
            'startup_times': dict(self.startup_times),
        }

    def __setstate__(self, state: dict) -> None:
        object.__setattr__(self, 'idle_timeout', state['idle_timeout'])
        object.__setattr__(self, 'cleanup_interval', state['cleanup_interval'])
        object.__setattr__(self, 'startup_times', MappingProxyType(state['startup_times']))
