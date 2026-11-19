"""Lettura dei parametri di scenario da formati diversi."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
import tomllib
from typing import TextIO

from sim.validation import finite_time


@dataclass(frozen=True)
class ScenarioConfig:
    first_event_time: float

    def __post_init__(self) -> None:
        if finite_time(self.first_event_time, 'first_event_time') < 0:
            raise ValueError('first_event_time deve essere non negativo')


class Scenario(ABC):
    """Legge uno scenario e applica le stesse regole a tutti i formati."""

    def load(self, path: str | Path) -> ScenarioConfig:
        with Path(path).open(encoding='utf-8') as source:
            data = self._read(source)
        if not isinstance(data, dict) or data.keys() != {'first_event_time'}:
            raise ValueError('Lo scenario deve contenere il solo parametro first_event_time')
        return ScenarioConfig(first_event_time=data['first_event_time'])

    @abstractmethod
    def _read(self, source: TextIO) -> object:
        """Decodifica il formato del file."""


class YamlScenario(Scenario):
    def _read(self, source: TextIO) -> object:
        import yaml

        try:
            return yaml.safe_load(source)
        except yaml.YAMLError as error:
            raise ValueError(f'YAML non valido: {error}') from error


class TomlScenario(Scenario):
    def _read(self, source: TextIO) -> object:
        return tomllib.loads(source.read())


class JsonScenario(Scenario):
    def _read(self, source: TextIO) -> object:
        return json.load(source)


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Sceglie il lettore in base all'estensione del file."""
    path = Path(path)
    readers = {'.yaml': YamlScenario, '.yml': YamlScenario,
               '.toml': TomlScenario, '.json': JsonScenario}
    reader = readers.get(path.suffix.lower())
    if reader is None:
        raise ValueError(f'Formato scenario non supportato: {path.suffix or "senza estensione"}')
    return reader().load(path)
