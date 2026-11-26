#!/usr/bin/env python3
"""Genera il report Markdown e JSON dai risultati di `mutmut` (v3) per GitHub Actions."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from mutmut.__main__ import get_diff_for_mutant, status_by_exit_code

REPO_ROOT = Path(__file__).resolve().parents[1]
_DIFF_REMOVE_RE = re.compile(r'^-(?!--)(.*)$')
_DIFF_ADD_RE = re.compile(r'^\+(?!\+\+)(.*)$')


def _index_function_lines(source: str) -> dict[str, tuple[int, int]]:
    """Indicizza i range di riga `(start, end)` per funzioni e metodi (`Class.method` o `func`)."""
    tree = ast.parse(source)
    ranges: dict[str, tuple[int, int]] = {}

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            key = '.'.join([*self._stack, node.name])
            ranges[key] = (node.lineno, getattr(node, 'end_lineno', node.lineno) or node.lineno)
            self._stack.append(node.name)
            self.generic_visit(node)
            self._stack.pop()

        visit_AsyncFunctionDef = visit_FunctionDef

    _Visitor().visit(tree)
    return ranges


def _extract_symbol_from_key(mutant_key: str) -> str:
    """Estrae `Class.method` o `func` da una chiave mutmut v3 (`mod.xǁClsǁmeth__mutmut_1`)."""
    tail = mutant_key.rsplit('.x', 1)[-1]
    tail = re.sub(r'__mutmut_\d+$', '', tail)
    if tail.startswith('ǁ'):
        return tail.lstrip('ǁ').replace('ǁ', '.')
    return tail.lstrip('_')


def _resolve_line_and_summary(
    source_lines: list[str],
    func_ranges: dict[str, tuple[int, int]],
    mutant_key: str,
    diff_text: str,
) -> tuple[int, str]:
    """Determina la riga nel file sorgente originale e una sintesi compatta della mutazione."""
    removed = [m.group(1).strip() for line in diff_text.splitlines() if (m := _DIFF_REMOVE_RE.match(line))]
    added = [m.group(1).strip() for line in diff_text.splitlines() if (m := _DIFF_ADD_RE.match(line))]

    symbol = _extract_symbol_from_key(mutant_key)
    start_line, end_line = func_ranges.get(symbol, (1, len(source_lines)))

    resolved_line = start_line
    if removed:
        needle = removed[0]
        for lineno in range(start_line, min(end_line, len(source_lines)) + 1):
            if needle and needle in source_lines[lineno - 1]:
                resolved_line = lineno
                break

    before = removed[0] if removed else '?'
    after = added[0] if added else '?'
    return resolved_line, f'`{before}` -> `{after}`'


def _format_markdown(report: dict) -> str:
    """Formatta il riepilogo di mutation testing in Markdown per terminale e GitHub Step Summary."""
    lines = [
        '## Mutation Testing Report (`mutmut` + `pytest`)',
        '',
        f"- **Mutation Score**: **{report['mutation_score_pct']:.2f}%** "
        f"({report['killed']} killed, {report['timeout']} timeout, "
        f"{report['survived']} survived, {report['no_tests']} no-tests / {report['checked']} checked)",
        '',
        '| File | Checked | Killed | Timeout | Survived | No tests | Score |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for file_path, stats in report['by_file'].items():
        lines.append(
            f"| `{file_path}` | {stats['checked']} | {stats['killed']} | {stats['timeout']} "
            f"| {stats['survived']} | {stats['no_tests']} | {stats['mutation_score_pct']:.2f}% |"
        )

    survivors = report['survived_mutants']
    if survivors:
        lines.extend([
            '',
            f'### Mutanti sopravvissuti ({len(survivors)})',
            '',
            '| File | Linea | Mutante | Diff |',
            '| --- | ---: | --- | --- |',
        ])
        for item in survivors[:50]:
            lines.append(
                f"| `{item['file_path']}` | {item['line']} | `{item['symbol']}` | {item['summary']} |"
            )
        if len(survivors) > 50:
            lines.append(f'\n*...e altri {len(survivors) - 50} mutanti sopravvissuti in `mutation-report.json`.*')

    return '\n'.join(lines) + '\n'


def main() -> int:
    parser = argparse.ArgumentParser(description='Genera i report Markdown e JSON dai risultati di mutmut.')
    parser.add_argument('--mutants-dir', type=Path, default=REPO_ROOT / 'mutants')
    parser.add_argument('--json-out', type=Path, default=REPO_ROOT / 'mutation-report.json')
    parser.add_argument('--md-out', type=Path, default=REPO_ROOT / 'mutation-report.md')
    parser.add_argument('--min-score', type=float, default=0.0)
    args = parser.parse_args()

    meta_files = sorted(args.mutants_dir.rglob('*.py.meta'))
    by_file: dict[str, dict[str, int | float]] = {}
    totals = {'checked': 0, 'killed': 0, 'survived': 0, 'timeout': 0, 'no_tests': 0, 'other': 0}
    survivors: list[dict] = []

    for meta_path in meta_files:
        rel_path = str(meta_path.relative_to(args.mutants_dir)).removesuffix('.meta')
        source_path = REPO_ROOT / rel_path
        if not source_path.is_file():
            continue

        data = json.loads(meta_path.read_text(encoding='utf-8'))
        exit_codes: dict[str, int | None] = data.get('exit_code_by_key', {})
        source_text = source_path.read_text(encoding='utf-8')
        source_lines = source_text.splitlines()
        func_ranges = _index_function_lines(source_text)

        stats = {'checked': 0, 'killed': 0, 'survived': 0, 'timeout': 0, 'no_tests': 0, 'other': 0}
        for key, code in sorted(exit_codes.items()):
            status = status_by_exit_code[code]
            if status == 'not checked':
                continue
            stats['checked'] += 1
            if status == 'killed':
                stats['killed'] += 1
            elif status == 'survived':
                stats['survived'] += 1
                diff_text = get_diff_for_mutant(key, path=Path(rel_path))
                lineno, summary = _resolve_line_and_summary(source_lines, func_ranges, key, diff_text)
                survivors.append({
                    'mutant_id': key,
                    'file_path': rel_path,
                    'symbol': _extract_symbol_from_key(key),
                    'line': lineno,
                    'summary': summary,
                })
            elif status == 'timeout':
                stats['timeout'] += 1
            elif status == 'no tests':
                stats['no_tests'] += 1
            else:
                stats['other'] += 1

        if stats['checked'] == 0:
            continue
        for k in totals:
            totals[k] += stats[k]
        detected = stats['killed'] + stats['timeout']
        file_score = round(100.0 * detected / stats['checked'], 2) if stats['checked'] else 100.0
        by_file[rel_path] = {**stats, 'mutation_score_pct': file_score}

    detected_total = totals['killed'] + totals['timeout']
    score = round(100.0 * detected_total / totals['checked'], 2) if totals['checked'] else 100.0
    survivors.sort(key=lambda item: (item['file_path'], item['line'], item['mutant_id']))

    report = {
        **totals,
        'mutation_score_pct': score,
        'by_file': by_file,
        'survived_mutants': survivors,
    }
    markdown = _format_markdown(report)

    args.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    args.md_out.write_text(markdown, encoding='utf-8')

    print(markdown)
    return 1 if score < args.min_score else 0


if __name__ == '__main__':
    raise SystemExit(main())
