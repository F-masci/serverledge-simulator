"""Avvio di uno scenario da file."""

import argparse
from pathlib import Path

from .io.scenario import Scenario
from .core.simulation import Simulation
from .utils.validation import nonnegative_integer


def main() -> None:
    parser = argparse.ArgumentParser(description='Avvia uno scenario di arrivi a eventi discreti')
    parser.add_argument('scenario', type=Path, help='File YAML, TOML o JSON')
    parser.add_argument('--arrivals', type=int, default=0, help='Numero di arrivi da osservare (default: 0)')
    args = parser.parse_args()

    try:
        arrivals = nonnegative_integer(args.arrivals, '--arrivals')
        simulation = Simulation(Scenario.load(args.scenario))
        first_arrivals = simulation.start()
        while simulation.arrivals < arrivals and simulation.engine.step():
            pass
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f'Tempo corrente: {simulation.engine.now} s')
    first_time = min(arrival.time for arrival in first_arrivals)
    print(f'Arrivi iniziali schedulati: {len(first_arrivals)} (primo a {first_time} s)')
    if arrivals:
        print(f'Arrivi elaborati: {simulation.arrivals}')
        print(f'Prossimo evento: {simulation.engine.next_event_time} s')


if __name__ == '__main__':
    main()
