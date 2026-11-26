"""Compare a real replay with the simulator at the last scheduled arrival."""

import argparse
import json
import os
import webbrowser
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from numpy import percentile
from simulator.io.scenario import Scenario
from simulator.observer import Notification, RequestCompleted, RequestRejected, observer
from simulator.utils.report import _percent, _seconds, _table

from replay import sample_arrivals
from report_html import render_html_report


def _safe_cli_path(path: Path) -> Path:
    """Risolve il percorso canonico e verifica che resti nella directory di lavoro corrente."""
    resolved = os.path.realpath(path)
    base_dir = os.path.realpath(os.getcwd())
    if resolved != base_dir and not resolved.startswith(base_dir + os.sep):
        raise ValueError(f"Percorso fuori dalla directory consentita: {path}")
    return Path(resolved)


@dataclass(frozen=True)
class Outcome:
    function: str
    status: str
    response_s: float | None = None
    service_s: float | None = None
    warm: bool | None = None
    node: str | None = None
    region: str | None = None


def format_placement(outcomes: list[Outcome]) -> str:
    """Formatta la distribuzione dei nodi/regioni di esecuzione per le richieste completate."""
    counts = Counter(
        (item.node or "-", item.region or "-")
        for item in outcomes
        if item.status == "completed"
    )
    if not counts:
        return "nessuna"
    return ", ".join(f"{node} [{region}] ×{count}" for (node, region), count in counts.most_common())


def summarize(outcomes: list[Outcome]) -> tuple[object, ...]:
    counts = Counter(item.status for item in outcomes)
    completed = [item for item in outcomes if item.status == "completed"]
    durations = [item.response_s for item in completed if item.response_s is not None]
    services = [item.service_s for item in completed if item.service_s is not None]
    starts = [item.warm for item in completed if item.warm is not None]
    return (
        len(outcomes), counts["completed"], counts["rejected"], counts["pending"], counts["error"],
        _percent(starts.count(False) / len(starts) if starts else None),
        _seconds(mean(durations) if durations else None),
        _seconds(float(percentile(durations, 95)) if durations else None),
        _seconds(mean(services) if services else None),
    )


def _load_replay_rows(results_path: Path) -> dict[int, dict]:
    """Carica e valida le righe JSONL prodotte dal replay reale."""
    rows: dict[int, dict] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        sequence = row["sequence"]
        if sequence in rows:
            raise ValueError(f"Sequenza duplicata: {sequence}")
        rows[sequence] = row
    if not rows or set(rows) != set(range(1, len(rows) + 1)):
        raise ValueError("Il file deve contenere un record per ogni richiesta, numerate da 1")
    return rows


def _simulate_outcomes(scenario: Scenario, total_requests: int) -> tuple[list, list[Outcome]]:
    """Esegue la traccia sul simulatore e raccoglie gli esiti delle richieste."""
    simulated: Counter[str] = Counter()
    simulated_outcomes: list[Outcome] = []

    def record(event: Notification) -> None:
        if isinstance(event, RequestCompleted):
            node_name = event.node.id.key if event.node is not None else None
            fallback_region = event.region.name if getattr(event, "region", None) is not None else None
            region_name = event.node.region.name if event.node is not None else fallback_region
            simulated_outcomes.append(Outcome(
                event.function.name, "completed", event.response_time_s,
                event.service_time_s, event.request.warm, node_name, region_name,
            ))
            simulated[event.function.name] += 1
        elif isinstance(event, RequestRejected):
            simulated_outcomes.append(Outcome(event.function.name, "rejected"))
            simulated[event.function.name] += 1

    unsubscribe = observer.subscribe(record)
    try:
        arrivals = sample_arrivals(scenario, total_requests)
    finally:
        unsubscribe()

    functions = Counter(arrival.function for arrival in arrivals)
    for name, count in (functions - simulated).items():
        simulated_outcomes.extend(Outcome(name, "pending") for _ in range(count))
    return arrivals, simulated_outcomes


def _build_real_outcomes(arrivals: list, rows: dict[int, dict], cutoff: float) -> list[Outcome]:
    """Converte i record del replay reale in `Outcome` coerenti con il cutoff temporale."""
    real_outcomes: list[Outcome] = []
    for arrival in arrivals:
        row = rows[arrival.sequence]
        if (row.get("function"), row.get("arrival_s"), row.get("service_s")) != (
            arrival.function, arrival.arrival_s, arrival.service_s,
        ):
            raise ValueError(f"La richiesta #{arrival.sequence} differisce dalla traccia. Verifica scenario e seed.")

        end = row["sent_s"] + row["response_s"]
        if end > cutoff:
            real_outcomes.append(Outcome(arrival.function, "pending"))
        elif row["success"]:
            response = row["response"]
            real_outcomes.append(Outcome(
                arrival.function, "completed", row["response_s"],
                response.get("Duration"), response.get("IsWarmStart"),
                response.get("ExecutionNode"), response.get("ExecutionArea"),
            ))
        else:
            status = "rejected" if row["http_status"] == 429 else "error"
            real_outcomes.append(Outcome(arrival.function, status))
    return real_outcomes


def _print_comparison_tables(
    cutoff: float,
    functions: Counter[str],
    simulated_outcomes: list[Outcome],
    real_outcomes: list[Outcome],
    rows: dict[int, dict],
) -> None:
    """Stampa le tabelle testuali di confronto tra simulatore e sistema reale."""
    headers = ("Sistema", "Arrivi", "Completate", "Rifiutate", "Pendenti", "Errori",
               "Cold %", "Risposta media", "p95", "Servizio medio")
    print(f"Confronto a t = {_seconds(cutoff)} dall'inizio della traccia")
    print(_table("Invocazioni e latenza", headers, [
        ("Simulatore", *summarize(simulated_outcomes)), ("Reale", *summarize(real_outcomes)),
    ]))
    print()
    print(_table("Invocazioni per funzione", ("Funzione", *headers), (
        (name, system, *summarize([item for item in outcomes if item.function == name]))
        for name in sorted(functions)
        for system, outcomes in (("Simulatore", simulated_outcomes), ("Reale", real_outcomes))
    )))
    print()
    print(_table(
        "Placement e forwarding per funzione (richieste completate entro cutoff)",
        ("Funzione", "Simulatore (nodi / regioni)", "Reale (nodi / aree)"),
        (
            (
                name,
                format_placement([item for item in simulated_outcomes if item.function == name]),
                format_placement([item for item in real_outcomes if item.function == name]),
            )
            for name in sorted(functions)
        ),
    ))
    print(f'\nEsito finale del replay reale: {sum(row["success"] for row in rows.values())}/{len(rows)} riuscite.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="File JSONL prodotto da validate.sh replay (o run)")
    parser.add_argument("--scenario", type=Path,
                        default=Path(__file__).resolve().parents[1] / "scenarios/validation.toml")
    parser.add_argument("--html", type=Path, help="Salva un report grafico HTML autonomo al percorso indicato")
    parser.add_argument("--open", action="store_true", help="Apri automaticamente il report HTML nel browser")
    args = parser.parse_args()

    try:
        results_path = _safe_cli_path(args.results)
        scenario_path = _safe_cli_path(args.scenario)
        html_path = _safe_cli_path(args.html) if args.html else None
        rows = _load_replay_rows(results_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    scenario = Scenario.load(scenario_path)
    arrivals, simulated_outcomes = _simulate_outcomes(scenario, len(rows))
    if len(arrivals) != len(rows):
        parser.error("Il numero di arrivi del simulatore differisce dal file reale")

    cutoff = arrivals[-1].arrival_s
    try:
        real_outcomes = _build_real_outcomes(arrivals, rows, cutoff)
    except ValueError as error:
        parser.error(str(error))

    functions = Counter(arrival.function for arrival in arrivals)
    _print_comparison_tables(cutoff, functions, simulated_outcomes, real_outcomes, rows)

    if html_path is not None:
        html_content = render_html_report(
            scenario_path=scenario_path,
            results_path=results_path,
            seed=scenario.seed,
            cutoff_s=cutoff,
            arrivals=arrivals,
            rows=rows,
            simulated_outcomes=simulated_outcomes,
            real_outcomes=real_outcomes,
        )
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_content, encoding="utf-8")
        print(f"Report HTML salvato in: {html_path}")
        if args.open:
            webbrowser.open(html_path.as_uri())


if __name__ == "__main__":
    main()
