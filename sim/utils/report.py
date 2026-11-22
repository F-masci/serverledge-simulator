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


def _section(title: str, rows: Iterable[tuple[str, object]]) -> str:
    """Formatta una sezione chiave-valore allineata per etichetta."""
    fields = tuple(rows)
    width = max(len(label) for label, _ in fields)
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

    # Sezioni principali: riepilogo invocazioni, tempi medi e stato dei container.
    sections = [
        _section('Invocazioni', (
            ('Arrivi', total['arrivals']),
            ('Accettate', total['accepted']),
            ('Completate', total['completed']),
            ('Rifiutate', total['rejected']),
            ('Pendenti', total['pending']),
            ('  In coda', total['waiting']),
            ('  In transito', total['in_transit']),
            ('Accodate durante la replica', total['queued']),
        )),
        _section('Tempi medi', (
            ('Risposta', _seconds(total['mean_response_time_s'])),
            ('Servizio', _seconds(total['mean_service_time_s'])),
            ('Attesa in coda', _seconds(total['mean_queue_wait_s'])),
            ('Creazione container', _seconds(containers['mean_creation_time_s'])),
        )),
        _section('Container', (
            ('Cold start', total['cold_starts']),
            ('Warm start', total['warm_starts']),
            ('Invocazioni senza container', total['accepted_without_container']),
            ('In creazione', containers['creating']),
            ('Occupati', containers['busy']),
            ('Inattivi', containers['idle']),
            ('Creati', containers['created']),
            ('Creazioni fallite', containers['creation_failed']),
            ('Rimossi', containers['destroyed']),
        )),
    ]

    # Inserisce il dettaglio dei motivi di rifiuto solo se presenti.
    if total['rejections_by_reason']:
        sections.insert(1, _section('Motivi di rifiuto', sorted(total['rejections_by_reason'].items())))

    # Tabella di dettaglio per funzione, se ci sono invocazioni osservate.
    if metrics['by_function']:
        sections.append(_table(
            'Invocazioni per funzione',
            ('Funzione', 'Arrivi', 'Completate', 'Rifiutate', 'Pendenti', 'Risposta media'),
            (
                (name, stats['arrivals'], stats['completed'], stats['rejected'], stats['pending'],
                 _seconds(stats['mean_response_time_s']))
                for name, stats in sorted(metrics['by_function'].items())
            ),
        ))

    # Tabella di riepilogo per regione e, se presenti, gli offload interregionali.
    primary = metrics['primary_region']
    sections.append(_table(
        'Invocazioni per regione',
        ('Regione', 'Modalità', 'Nodi', 'Arrivi esterni', 'Completate', 'Rifiutate'),
        (
            (f'{name} (primaria)' if name == primary else name,
             'simulata' if stats['mode'] == 'simulated' else 'sintetica',
             sum(map(len, stats['nodes_by_ring'].values())), stats['external_arrivals'],
             stats['invocations']['completed'], stats['invocations']['rejected'])
            for name, stats in sorted(metrics['regions'].items())
        ),
    ))
    if metrics['regional_offloads']:
        sections.append(_table(
            'Offloading interregionale',
            ('Origine', 'Destinazione', 'Inoltrate'),
            ((row['source'], row['destination'], row['count']) for row in metrics['regional_offloads']),
        ))
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
