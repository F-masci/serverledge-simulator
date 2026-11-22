"""Formattazione del report finale della simulazione per il terminale."""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..observer import FinalStatistics


##########################
### Formatting helpers ###
##########################

def _seconds(value: float | None) -> str:
    """Formatta una durata in secondi o 'n.d.' se non disponibile."""
    return 'n.d.' if value is None else f'{value:.6g} s'


def _percent(value: float | None) -> str:
    """Formatta un rapporto in [0, 1] come percentuale o 'n.d.' se non disponibile."""
    return 'n.d.' if value is None else f'{value * 100:.2f}%'


def _number(value: float | None, suffix: str = '') -> str:
    """Formatta un numero reale con eventuale unità di misura o 'n.d.'."""
    if value is None:
        return 'n.d.'
    return f'{value:.4g}{suffix}'


def _section(title: str, rows: Iterable[tuple[str, object]]) -> str:
    """Formatta una sezione chiave-valore allineata per etichetta."""
    fields = tuple(rows)
    width = max((len(label) for label, _ in fields), default=0)
    return '\n'.join([title, *(f'  {label:<{width}}  {value}' for label, value in fields)])


def _table(title: str, headers: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    """Formatta una tabella testuale allineando a destra le colonne puramente numeriche."""
    cells = [headers, *(tuple(map(str, row)) for row in rows)]
    columns = tuple(zip(*cells, strict=True))
    widths = [max(map(len, column)) for column in columns]
    numeric = [all(value.isdecimal() for value in column[1:]) for column in columns]
    return '\n'.join([
        title,
        *(
            '  ' + '  '.join(
                value.rjust(width) if align_right else value.ljust(width)
                for value, width, align_right in zip(row, widths, numeric, strict=True)
            ).rstrip()
            for row in cells
        ),
    ])


#######################
### Report builders ###
#######################

def format_metrics(metrics: dict) -> str:
    """Organizza lo snapshot in sezioni leggibili senza ricalcolare le metriche."""
    total = metrics['total']
    containers = total['containers']
    rt_pct = total['response_time_percentiles_s']

    # Sezioni principali: riepilogo invocazioni, tempi/percentili e stato dei container.
    in_execution = total['accepted'] - total['completed']
    sections = [
        _section('Invocazioni', (
            ('Arrivi', total['arrivals']),
            ('Accettate', f"{total['accepted']} ({_percent(total['acceptance_rate'])})"),
            ('Completate', total['completed']),
            ('Rifiutate', f"{total['rejected']} ({_percent(total['rejection_rate'])})"),
            ('Pendenti', total['pending']),
            ('  In esecuzione', in_execution),
            ('  In coda', total['waiting']),
            ('  In transito', total['in_transit']),
            ('Accodate durante la replica', total['queued']),
            ('Lunghezza media / max coda', f"{_number(total['mean_queue_length'])} / {total['max_queue_length']}"),
            ('Tasso di offload', _percent(total.get('offload_rate'))),
            ('Throughput', _number(total['throughput_rps'], ' req/s')),
        )),
        _section('Tempi e latenze', (
            ('Risposta media', _seconds(total['mean_response_time_s'])),
            ('  Min', _seconds(rt_pct['min'])),
            ('  p50', _seconds(rt_pct['p50'])),
            ('  p90', _seconds(rt_pct['p90'])),
            ('  p95', _seconds(rt_pct['p95'])),
            ('  p99', _seconds(rt_pct['p99'])),
            ('  Max', _seconds(rt_pct['max'])),
            ('Servizio medio', _seconds(total['mean_service_time_s'])),
            ('Attesa in coda media', _seconds(total['mean_queue_wait_s'])),
            ('Latenza di rete media', _seconds(total['mean_network_latency_s'])),
            ('Creazione container media', _seconds(containers['mean_creation_time_s'])),
        )),
        _section('Container', (
            ('Cold start', f"{total['cold_starts']} ({_percent(total['cold_start_rate'])})"),
            ('Warm start', f"{total['warm_starts']} ({_percent(total['warm_start_rate'])})"),
            ('Invocazioni senza container', total['accepted_without_container']),
            ('In creazione', containers['creating']),
            ('Occupati', containers['busy']),
            ('Inattivi', containers['idle']),
            ('Creati', containers['created']),
            ('Creazioni fallite', containers['creation_failed']),
            ('Rimossi', containers['destroyed']),
            ('  Scaduti (TTL)', containers['expired']),
            ('  Evictati (RAM)', containers['evicted']),
            ('Attivazioni medie per container', _number(containers['mean_activations_per_container'])),
            ('Permanenza media in idle', _seconds(containers['mean_idle_duration_s'])),
        )),
    ]

    # Inserisce il dettaglio dei motivi di rifiuto solo se presenti.
    if total['rejections_by_reason']:
        sections.insert(1, _section('Motivi di rifiuto', sorted(total['rejections_by_reason'].items())))

    # Tabella di dettaglio per funzione, se ci sono invocazioni osservate.
    if metrics['by_function']:
        sections.append(_table(
            'Invocazioni per funzione',
            ('Funzione', 'Arrivi', 'Completate', 'Rifiutate', 'Pendenti', 'Cold %', 'Risposta media', 'p95'),
            (
                (
                    name,
                    stats['arrivals'],
                    stats['completed'],
                    stats['rejected'],
                    stats['pending'],
                    _percent(stats['cold_start_rate']),
                    _seconds(stats['mean_response_time_s']),
                    _seconds(stats['response_time_percentiles_s']['p95']),
                )
                for name, stats in sorted(metrics['by_function'].items())
            ),
        ))

    # Tabella di riepilogo per regione.
    primary = metrics['primary_region']
    sections.append(_table(
        'Invocazioni per regione',
        ('Regione', 'Modalità', 'Nodi', 'Arrivi esterni', 'Completate', 'Rifiutate', 'CPU media', 'RAM busy'),
        (
            (
                f'{name} (primaria)' if name == primary else name,
                'simulata' if stats['mode'] == 'simulated' else 'sintetica',
                sum(map(len, stats['nodes_by_ring'].values())),
                stats['external_arrivals'],
                stats['invocations']['completed'],
                stats['invocations']['rejected'],
                _percent(stats['resources']['mean_cpu_utilization']),
                _percent(stats['resources']['mean_busy_memory_utilization']),
            )
            for name, stats in sorted(metrics['regions'].items())
        ),
    ))

    # Tabella di dettaglio per ring (aggregazione per architettura / tag).
    ring_rows = [
        (
            region_name,
            ring_id,
            ring_stats['nodes_count'],
            ring_stats['invocations']['accepted'],
            ring_stats['invocations']['rejected'],
            _percent(ring_stats['invocations']['cold_start_rate']),
            _percent(ring_stats['resources']['mean_cpu_utilization']),
            _percent(ring_stats['resources']['mean_busy_memory_utilization']),
            _number(ring_stats['load_balance_ratio']),
            _seconds(ring_stats['invocations']['mean_response_time_s']),
        )
        for region_name, region_stats in sorted(metrics['regions'].items())
        for ring_id, ring_stats in sorted(region_stats.get('rings', {}).items())
    ]
    if ring_rows:
        sections.append(_table(
            'Metriche per ring',
            ('Regione', 'Ring', 'Nodi', 'Accettate', 'Rifiutate', 'Cold %', 'CPU media', 'RAM busy', 'Bilanc.', 'Risposta media'),
            ring_rows,
        ))

    # Tabelle di offloading interregionale e intra-regione, se presenti.
    if metrics['regional_offloads']:
        sections.append(_table(
            'Offloading interregionale',
            ('Origine', 'Destinazione', 'Inoltrate'),
            ((row['source'], row['destination'], row['count']) for row in metrics['regional_offloads']),
        ))
    if metrics.get('intra_region_offloads'):
        sections.append(_table(
            'Offloading intra-regione',
            ('Regione', 'Origine', 'Destinazione', 'Inoltrate'),
            (
                (row['region'], row['source'], row['destination'], row['count'])
                for row in metrics['intra_region_offloads']
            ),
        ))

    # Sezione topologia e capacità allocata nel tempo.
    topology = metrics.get('topology')
    if topology is not None:
        sections.append(_section('Topologia e capacità', (
            ('Nodi attivi (correnti / media)', f"{topology['current_active_nodes']} / {_number(topology['mean_active_nodes'])}"),
            ('Eventi di scaling (out / in)', f"{topology['scale_out_events']} / {topology['scale_in_events']}"),
            ('CPU·s allocati', _number(topology['cpu_seconds_allocated'])),
            ('RAM·s allocati (MB·s)', _number(topology['memory_mb_seconds_allocated'])),
        )))

    return '\n\n'.join(sections)


def format_final_report(statistics: 'FinalStatistics', scenario: Path) -> str:
    """Formatta il report della CLI a partire dalla notifica finale della replica."""
    outcomes = {'completed': 'Terminata', 'interrupted': 'Interrotta', 'failed': 'Fallita'}
    header = _section('Riepilogo simulazione', (
        ('Scenario', scenario),
        ('Replica', statistics.run_id),
        ('Esito', outcomes[statistics.outcome]),
        ('Tempo simulato', _seconds(statistics.sim_time_s)),
    ))
    return f'{header}\n\n{statistics.summary}'
