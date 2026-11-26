"""Replay the simulator's sampled workload against the local Serverledge lab."""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import TextIO
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from simulator.core.simulation import Simulation
from simulator.io.scenario import Scenario
from simulator.observer import Notification, RequestArrived, observer


def _safe_cli_path(path: Path) -> Path:
    """Risolve il percorso canonico e verifica che resti nella directory di lavoro corrente."""
    resolved = os.path.realpath(path)
    base_dir = os.path.realpath(os.getcwd())
    if resolved != base_dir and not resolved.startswith(base_dir + os.sep):
        raise ValueError(f"Percorso fuori dalla directory consentita: {path}")
    return Path(resolved)


@dataclass(frozen=True)
class Arrival:
    sequence: int
    function: str
    arrival_s: float
    service_s: float


def sample_arrivals(scenario: Scenario, count: int) -> list[Arrival]:
    arrivals: list[Arrival] = []
    simulation = Simulation(scenario)

    def record(event: Notification) -> None:
        if isinstance(event, RequestArrived) and event.run_id == simulation.run_id:
            duration = event.request.service_duration
            assert duration is not None
            arrivals.append(Arrival(len(arrivals) + 1, event.function.name, event.sim_time_s, duration))

    unsubscribe = observer.subscribe(record)
    try:
        simulation.run(arrivals=count)
    finally:
        unsubscribe()
    return arrivals


def invoke(arrival: Arrival, url: str, can_offload: bool, timeout: float, origin: float) -> dict:
    data = json.dumps({
        "CanDoOffloading": can_offload,
        "Params": {"service_s": arrival.service_s},
    }).encode()
    request = Request(f"{url}/invoke/{quote('sim_' + arrival.function, safe='')}", data,
                      {"Content-Type": "application/json"})
    started = monotonic()
    result = {**asdict(arrival), "sent_s": started - origin, "http_status": None, "success": False}
    try:
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            result["http_status"] = response.status
            body = response.read().decode(errors="replace")
        try:
            result["response"] = json.loads(body)
        except json.JSONDecodeError:
            result["response"] = body
        result["success"] = (
            result["http_status"] == 200 and isinstance(result["response"], dict)
            and result["response"].get("Success") is True
        )
    except OSError as error:
        result["error"] = str(error)
    result["response_s"] = monotonic() - started
    return result


def replay(
    arrivals: list[Arrival], args: argparse.Namespace, scenario: Scenario, output: TextIO | None,
) -> None:
    origin = monotonic()
    display_lock = Lock()

    def send(arrival: Arrival) -> bool:
        result = invoke(arrival, args.url.rstrip("/"), scenario.invocation.can_offload, args.timeout, origin)
        response = result.get("response")
        destination = response.get("ExecutionNode") if isinstance(response, dict) else None
        status = result["http_status"] if result["http_status"] is not None else result.get("error")
        with display_lock:
            print(f'← #{arrival.sequence} {arrival.function} {status} '
                  f'{"OK" if result["success"] else "KO"} {result["response_s"]:.3f}s '
                  f'nodo={destination or "-"}', flush=True)
            if output is not None:
                print(json.dumps(result), file=output, flush=True)
        return result["success"]

    pending = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for arrival in arrivals:
            sleep(max(0, origin + arrival.arrival_s - monotonic()))
            print(f'→ #{arrival.sequence} {arrival.function} t={arrival.arrival_s:.3f}s '
                  f'servizio={arrival.service_s:.3f}s', flush=True)
            pending.append(pool.submit(send, arrival))
        successes = sum(task.result() for task in pending)
    print(f'Completate con successo: {successes}/{len(arrivals)}. Fallite: {len(arrivals) - successes}.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path,
                        default=Path(__file__).resolve().parents[1] / "scenarios/validation.toml")
    parser.add_argument("--arrivals", type=int, default=100)
    parser.add_argument("--url", default="http://127.0.0.1:13231")
    parser.add_argument("--concurrency", type=int, default=32, help="Massimo di richieste HTTP in volo")
    parser.add_argument("--timeout", type=float, default=30, help="Timeout HTTP in secondi")
    parser.add_argument("--output", type=Path, help="Salva le risposte in JSONL")
    parser.add_argument("--dry-run", action="store_true", help="Stampa la traccia JSONL senza inviare richieste")
    args = parser.parse_args()
    if args.arrivals < 1 or args.concurrency < 1 or args.timeout <= 0:
        parser.error("--arrivals, --concurrency e --timeout devono essere positivi")

    try:
        scenario_path = _safe_cli_path(args.scenario)
        output_path = _safe_cli_path(args.output) if args.output else None
    except ValueError as error:
        parser.error(str(error))

    scenario = Scenario.load(scenario_path)
    if not args.dry_run and (scenario.invocation.entry_node is not None
                             or scenario.invocation.entry_region not in (None, scenario.primary_region)):
        parser.error("Il replay HTTP usa il bilanciatore della regione primaria. Usa --dry-run per altri ingressi.")

    arrivals = sample_arrivals(scenario, args.arrivals)
    default_stream = sys.stdout if args.dry_run else None
    try:
        output = output_path.open("x") if output_path else nullcontext(default_stream)
    except OSError as error:
        parser.error(str(error))
    with output as stream:
        if args.dry_run:
            for arrival in arrivals:
                print(json.dumps(asdict(arrival)), file=stream)
        else:
            replay(arrivals, args, scenario, stream)


if __name__ == "__main__":
    main()
