"""Avvio di uno scenario da file."""

import argparse
import logging
import os
from pathlib import Path

from .core.simulation import Simulation
from .io.scenario import Scenario
from .observer import SimulationError, observer
from .utils.logger import log_notification
from .utils.report import format_final_report
from .utils.validation import nonnegative_integer


def _resolve_scenario_path(path: Path) -> Path:
    """Risolve il percorso canonico e verifica che resti nella directory di lavoro corrente."""
    resolved = os.path.realpath(path)
    base_dir = os.path.realpath(os.getcwd())
    if resolved != base_dir and not resolved.startswith(base_dir + os.sep):
        raise ValueError(f'Percorso scenario fuori dalla directory consentita: {path}')
    return Path(resolved)


def main() -> None:
    """Punto di ingresso CLI per eseguire uno scenario e stampare il riepilogo."""

    ### Parsing ###
    parser = argparse.ArgumentParser(description='Avvia uno scenario di arrivi a eventi discreti')
    parser.add_argument('scenario', type=Path, help='File YAML, TOML o JSON')
    parser.add_argument('--arrivals', type=int, default=0, help='Numero di arrivi da osservare (default: 0)')
    parser.add_argument('--log-level', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'), default='WARNING', help='Livello minimo del logger (default: WARNING)')
    args = parser.parse_args()

    ### Validation ###
    try:
        nonnegative_integer(args.arrivals, 'arrivals')
    except ValueError as error:
        parser.error(str(error))

    ### Logging ###
    logging.basicConfig(level=args.log_level, format='%(message)s')
    unsubscribe_logger = observer.subscribe(log_notification)

    ### Scenario ###
    try:
        scenario_path = _resolve_scenario_path(args.scenario)
        scenario = Scenario.load(scenario_path)
    except Exception as error:
        observer.notify(SimulationError(exception=error, failed_event='Scenario.load'))
        unsubscribe_logger()
        raise SystemExit(2) from error

    ### Simulation Loop ###
    try:
        simulation = Simulation(scenario)
        statistics = simulation.run(arrivals=args.arrivals)
    except Exception as error:
        observer.notify(SimulationError(exception=error, failed_event='Simulation Loop'))
        raise SystemExit(2) from error
    finally:
        unsubscribe_logger()

    observer.notify(statistics)
    print(format_final_report(statistics, args.scenario))

if __name__ == '__main__':
    main()
