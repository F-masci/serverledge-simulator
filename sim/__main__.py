"""Avvio di uno scenario da file."""

import argparse
from pathlib import Path

from .io.scenario import load_scenario
from .simulation import Simulation


def main() -> None:
    parser = argparse.ArgumentParser(description='Schedula il primo evento dello scenario')
    parser.add_argument('scenario', type=Path, help='File YAML, TOML o JSON')
    args = parser.parse_args()

    try:
        simulation = Simulation(load_scenario(args.scenario))
        first_event = simulation.start()
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f'Tempo corrente: {simulation.engine.now} s')
    print(f'Primo evento schedulato a {first_event.time} s')


if __name__ == '__main__':
    main()
