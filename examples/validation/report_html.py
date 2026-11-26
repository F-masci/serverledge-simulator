"""Generatore di report HTML interattivo per il confronto simulatore vs Serverledge reale."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from statistics import mean
from typing import Any

from numpy import percentile


@dataclass(frozen=True)
class MetricSummary:
    """Metriche aggregate numeriche per un sistema o una singola funzione."""

    arrivals: int
    completed: int
    rejected: int
    pending: int
    errors: int
    cold_pct: float | None
    mean_response_s: float | None
    p95_response_s: float | None
    mean_service_s: float | None


def compute_metrics(outcomes: list[Any]) -> MetricSummary:
    """Calcola le metriche aggregate a partire da una lista di Outcome."""
    counts = Counter(item.status for item in outcomes)
    completed = [item for item in outcomes if item.status == "completed"]
    durations = [item.response_s for item in completed if item.response_s is not None]
    services = [item.service_s for item in completed if item.service_s is not None]
    starts = [item.warm for item in completed if item.warm is not None]
    return MetricSummary(
        arrivals=len(outcomes),
        completed=counts["completed"],
        rejected=counts["rejected"],
        pending=counts["pending"],
        errors=counts["error"],
        cold_pct=(100.0 * starts.count(False) / len(starts)) if starts else None,
        mean_response_s=mean(durations) if durations else None,
        p95_response_s=float(percentile(durations, 95)) if durations else None,
        mean_service_s=mean(services) if services else None,
    )


def _fmt_sec(value: float | None) -> str:
    if value is None:
        return "n.d."
    return f"{value:.3f} s"


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n.d."
    return f"{value:.1f}%"


def _fmt_delta(sim: float | None, real: float | None, unit: str = "s") -> str:
    if sim is None or real is None:
        return '<span class="delta muted">n.d.</span>'
    diff = real - sim
    sign = "+" if diff > 0 else ""
    if unit == "s":
        text = f"{sign}{diff * 1000:.1f} ms"
        pct = f" ({sign}{(diff / sim) * 100:.1f}%)" if sim > 1e-9 else ""
        cls = "neutral" if abs(diff) < 0.05 else ("warn" if diff > 0 else "good")
        return f'<span class="delta {cls}">{escape(text + pct)}</span>'
    if unit == "%":
        text = f"{sign}{diff:.1f} pp"
        cls = "neutral" if abs(diff) < 5.0 else ("warn" if diff > 0 else "good")
        return f'<span class="delta {cls}">{escape(text)}</span>'
    text = f"{sign}{int(round(diff))}"
    cls = "neutral" if diff == 0 else "info"
    return f'<span class="delta {cls}">{escape(text)}</span>'


def _svg_stacked_outcomes(sim: MetricSummary, real: MetricSummary) -> str:
    """Grafico SVG a barre orizzontali impilate per gli esiti delle richieste."""
    width = 560
    height = 148
    bar_x = 112
    bar_w = 370
    bar_h = 30
    segments = [
        ("completed", "Completate", "#10b981"),
        ("rejected", "Rifiutate", "#ef4444"),
        ("pending", "Pendenti", "#f59e0b"),
        ("errors", "Errori", "#6366f1"),
    ]
    rows = [("Simulatore", sim, 24), ("Reale", real, 72)]
    parts = [f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Distribuzione esiti">']
    for label, summary, y in rows:
        total = max(summary.arrivals, 1)
        parts.append(
            f'<text x="{bar_x - 12}" y="{y + 20}" text-anchor="end" class="svg-label-bold">{escape(label)}</text>'
        )
        parts.append(f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="6" fill="#e2e8f0" />')
        x = float(bar_x)
        for attr, _name, color in segments:
            val = getattr(summary, attr)
            if val <= 0:
                continue
            seg_w = bar_w * (val / total)
            parts.append(
                f'<rect x="{x:.1f}" y="{y}" width="{seg_w:.1f}" height="{bar_h}" fill="{color}">'
                f'<title>{escape(label)} - {escape(_name)}: {val}/{summary.arrivals}</title></rect>'
            )
            if seg_w >= 26:
                parts.append(
                    f'<text x="{x + seg_w / 2:.1f}" y="{y + 19}" text-anchor="middle" '
                    f'class="svg-inbar">{val}</text>'
                )
            x += seg_w
        parts.append(
            f'<text x="{bar_x + bar_w + 10}" y="{y + 20}" class="svg-muted">{summary.completed}/{summary.arrivals} ok</text>'
        )

    # Legenda
    lx = bar_x
    ly = 128
    for _attr, name, color in segments:
        parts.append(f'<rect x="{lx}" y="{ly - 10}" width="11" height="11" rx="2" fill="{color}" />')
        parts.append(f'<text x="{lx + 16}" y="{ly}" class="svg-legend">{escape(name)}</text>')
        lx += 94
    parts.append("</svg>")
    return "".join(parts)


def _svg_grouped_bars(
    items: list[tuple[str, float | None, float | None]],
    unit: str,
    max_override: float | None = None,
) -> str:
    """Grafico SVG a barre raggruppate (Simulatore vs Reale) per funzione."""
    width = 560
    height = 225
    margin_l, margin_r, margin_t, margin_b = 54, 18, 28, 42
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    values = [v for _, s, r in items for v in (s, r) if v is not None]
    max_val = max_override if max_override is not None else (max(values) * 1.18 if values else 1.0)
    if max_val <= 0:
        max_val = 1.0

    parts = [f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img">']
    # Legenda in alto
    parts.append(f'<rect x="{margin_l}" y="6" width="11" height="11" rx="2" fill="#2563eb" />')
    parts.append(f'<text x="{margin_l + 16}" y="15" class="svg-legend">Simulatore</text>')
    parts.append(f'<rect x="{margin_l + 96}" y="6" width="11" height="11" rx="2" fill="#f97316" />')
    parts.append(f'<text x="{margin_l + 112}" y="15" class="svg-legend">Reale</text>')

    # Griglia orizzontale
    ticks = 4
    for i in range(ticks + 1):
        frac = i / ticks
        y = margin_t + plot_h * (1 - frac)
        val = max_val * frac
        label = f"{val:.0f}%" if unit == "%" else f"{val:.2f}"
        parts.append(f'<line x1="{margin_l}" y1="{y:.1f}" x2="{width - margin_r}" y2="{y:.1f}" class="svg-grid" />')
        parts.append(f'<text x="{margin_l - 8}" y="{y + 4:.1f}" text-anchor="end" class="svg-axis">{label}</text>')

    count = max(len(items), 1)
    group_w = plot_w / count
    bar_w = min(24.0, group_w * 0.32)
    gap = 4.0

    for idx, (name, sim_v, real_v) in enumerate(items):
        cx = margin_l + group_w * (idx + 0.5)
        for offset, val, color, sys_name in (
            (-(bar_w + gap / 2), sim_v, "#2563eb", "Simulatore"),
            (gap / 2, real_v, "#f97316", "Reale"),
        ):
            bx = cx + offset
            if val is None:
                parts.append(
                    f'<text x="{bx + bar_w / 2:.1f}" y="{margin_t + plot_h - 4:.1f}" '
                    f'text-anchor="middle" class="svg-na">n.d.</text>'
                )
            else:
                bh = max(2.0, (val / max_val) * plot_h)
                by = margin_t + plot_h - bh
                fmt = f"{val:.0f}%" if unit == "%" else f"{val:.2f}s"
                parts.append(
                    f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" rx="3" fill="{color}">'
                    f'<title>{escape(name)} ({sys_name}): {fmt}</title></rect>'
                )
                parts.append(
                    f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" class="svg-val">{fmt}</text>'
                )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 16}" text-anchor="middle" class="svg-label">{escape(name)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def _svg_node_distribution(node_counts: list[tuple[str, str, int]]) -> str:
    """Grafico SVG a barre orizzontali dei nodi di esecuzione nel sistema reale."""
    if not node_counts:
        return '<p class="empty-msg">Nessuna invocazione completata con ExecutionNode registrato.</p>'
    width = 560
    row_h = 30
    height = max(90, 24 + len(node_counts) * row_h)
    bar_x = 148
    bar_w = 320
    max_c = max(c for _, _, c in node_counts)
    area_colors = {
        "SIM_EDGE": "#0ea5e9",
        "SIM_REGIONAL": "#8b5cf6",
        "SIM_CLOUD": "#14b8a6",
    }
    parts = [f'<svg class="chart-svg" viewBox="0 0 {width} {height}" role="img" aria-label="Nodi di esecuzione reali">']
    for idx, (node, area, count) in enumerate(node_counts):
        y = 14 + idx * row_h
        w = max(4.0, (count / max_c) * bar_w)
        color = area_colors.get(area, "#64748b")
        parts.append(f'<text x="{bar_x - 10}" y="{y + 15}" text-anchor="end" class="svg-label-bold">{escape(node)}</text>')
        parts.append(f'<rect x="{bar_x}" y="{y + 2}" width="{bar_w}" height="18" rx="4" fill="#f1f5f9" />')
        parts.append(
            f'<rect x="{bar_x}" y="{y + 2}" width="{w:.1f}" height="18" rx="4" fill="{color}">'
            f'<title>{escape(node)} ({escape(area)}): {count} richieste</title></rect>'
        )
        parts.append(
            f'<text x="{bar_x + bar_w + 10}" y="{y + 15}" class="svg-muted">{count} ({escape(area)})</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _format_placement_pills(outcomes: list[Any], *, pill_cls: str) -> str:
    """Costruisce i badge HTML nodo [regione/area] ×N per le richieste completate."""
    counts = Counter(
        (getattr(item, "node", None) or "-", getattr(item, "region", None) or "-")
        for item in outcomes
        if item.status == "completed"
    )
    if not counts:
        return '<span class="muted">nessuna completata</span>'
    pills = []
    for (node, region), count in counts.most_common():
        pills.append(
            f'<span class="pill {pill_cls}">{escape(node)} '
            f'<span class="pill-region">[{escape(region)}]</span> ×{count}</span>'
        )
    return '<div class="pill-list">' + "".join(pills) + "</div>"


def render_html_report(
    *,
    scenario_path: Path,
    results_path: Path,
    seed: int,
    cutoff_s: float,
    arrivals: list[Any],
    rows: dict[int, dict[str, Any]],
    simulated_outcomes: list[Any],
    real_outcomes: list[Any],
) -> str:
    """Costruisce una pagina HTML autonoma con CSS e grafici SVG integrati."""
    sim_total = compute_metrics(simulated_outcomes)
    real_total = compute_metrics(real_outcomes)
    fn_names = sorted({item.function for item in arrivals})
    per_fn = {
        name: (
            compute_metrics([o for o in simulated_outcomes if o.function == name]),
            compute_metrics([o for o in real_outcomes if o.function == name]),
        )
        for name in fn_names
    }

    # Distribuzione ExecutionNode dal replay reale
    node_area_counter: Counter[tuple[str, str]] = Counter()
    for row in rows.values():
        resp = row.get("response")
        if row.get("success") and isinstance(resp, dict) and resp.get("ExecutionNode"):
            node_area_counter[(str(resp["ExecutionNode"]), str(resp.get("ExecutionArea") or "-"))] += 1
    node_counts = [(node, area, count) for (node, area), count in node_area_counter.most_common()]

    real_final_ok = sum(1 for row in rows.values() if row.get("success"))
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Grafici SVG
    chart_outcomes = _svg_stacked_outcomes(sim_total, real_total)
    chart_resp = _svg_grouped_bars([(n, per_fn[n][0].mean_response_s, per_fn[n][1].mean_response_s) for n in fn_names], "s")
    chart_p95 = _svg_grouped_bars([(n, per_fn[n][0].p95_response_s, per_fn[n][1].p95_response_s) for n in fn_names], "s")
    chart_cold = _svg_grouped_bars(
        [(n, per_fn[n][0].cold_pct, per_fn[n][1].cold_pct) for n in fn_names], "%", max_override=100.0
    )
    chart_service = _svg_grouped_bars([(n, per_fn[n][0].mean_service_s, per_fn[n][1].mean_service_s) for n in fn_names], "s")
    chart_nodes = _svg_node_distribution(node_counts)

    # Righe tabella per funzione
    fn_rows_html = []
    for name in fn_names:
        s, r = per_fn[name]
        fn_rows_html.append(
            f"""
            <tr class="fn-group-start">
              <td rowspan="2" class="fn-name">{escape(name)}</td>
              <td><span class="pill pill-sim">Simulatore</span></td>
              <td class="num">{s.arrivals}</td>
              <td class="num">{s.completed}</td>
              <td class="num">{s.rejected}</td>
              <td class="num">{s.pending}</td>
              <td class="num">{s.errors}</td>
              <td class="num">{_fmt_pct(s.cold_pct)}</td>
              <td class="num">{_fmt_sec(s.mean_response_s)}</td>
              <td class="num">{_fmt_sec(s.p95_response_s)}</td>
              <td class="num">{_fmt_sec(s.mean_service_s)}</td>
              <td rowspan="2" class="num">{_fmt_delta(s.mean_response_s, r.mean_response_s, 's')}</td>
            </tr>
            <tr>
              <td><span class="pill pill-real">Reale</span></td>
              <td class="num">{r.arrivals}</td>
              <td class="num">{r.completed}</td>
              <td class="num">{r.rejected}</td>
              <td class="num">{r.pending}</td>
              <td class="num">{r.errors}</td>
              <td class="num">{_fmt_pct(r.cold_pct)}</td>
              <td class="num">{_fmt_sec(r.mean_response_s)}</td>
              <td class="num">{_fmt_sec(r.p95_response_s)}</td>
              <td class="num">{_fmt_sec(r.mean_service_s)}</td>
            </tr>"""
        )

    # Righe tabella placement e forwarding (Simulatore vs Reale)
    placement_rows_html = []
    for name in fn_names:
        sim_fn = [o for o in simulated_outcomes if o.function == name]
        real_fn = [o for o in real_outcomes if o.function == name]
        sim_nodes = {getattr(o, "node", None) for o in sim_fn if o.status == "completed" and getattr(o, "node", None)}
        real_nodes = {getattr(o, "node", None) for o in real_fn if o.status == "completed" and getattr(o, "node", None)}
        if not real_nodes and not sim_nodes:
            match_badge = '<span class="delta muted">n.d.</span>'
        elif sim_nodes == real_nodes:
            match_badge = '<span class="delta good">Allineato</span>'
        else:
            match_badge = '<span class="delta warn">Routing diverso</span>'
        placement_rows_html.append(
            f"""
            <tr>
              <td class="fn-name">{escape(name)}</td>
              <td>{_format_placement_pills(sim_fn, pill_cls="pill-sim")}</td>
              <td>{_format_placement_pills(real_fn, pill_cls="pill-real")}</td>
              <td>{match_badge}</td>
            </tr>"""
        )

    # Righe dettaglio richieste reali
    req_rows_html = []
    for arrival in arrivals:
        row = rows[arrival.sequence]
        end = row["sent_s"] + row["response_s"]
        resp = row.get("response") if isinstance(row.get("response"), dict) else {}
        if end > cutoff_s:
            status_key, status_label = "pending", "Pendente"
        elif row.get("success"):
            status_key, status_label = "completed", "Completata"
        elif row.get("http_status") == 429:
            status_key, status_label = "rejected", "Rifiutata (429)"
        else:
            status_key, status_label = "error", "Errore"

        warm_val = resp.get("IsWarmStart")
        start_badge = (
            '<span class="pill pill-warm">Warm</span>'
            if warm_val is True
            else ('<span class="pill pill-cold">Cold</span>' if warm_val is False else '<span class="muted">-</span>')
        )
        node = str(resp.get("ExecutionNode") or "-")
        area = str(resp.get("ExecutionArea") or "-")
        http_code = str(row.get("http_status") if row.get("http_status") is not None else row.get("error", "-"))

        req_rows_html.append(
            f"""
            <tr data-fn="{escape(arrival.function)}" data-status="{status_key}">
              <td class="num">#{arrival.sequence}</td>
              <td><strong>{escape(arrival.function)}</strong></td>
              <td class="num">{arrival.arrival_s:.3f}s</td>
              <td class="num">{row['sent_s']:.3f}s</td>
              <td class="num">{arrival.service_s:.3f}s</td>
              <td class="num">{row['response_s']:.3f}s</td>
              <td>{start_badge}</td>
              <td>{escape(node)}</td>
              <td><span class="muted">{escape(area)}</span></td>
              <td class="num">{escape(http_code)}</td>
              <td><span class="pill pill-{status_key}">{status_label}</span></td>
            </tr>"""
        )

    fn_filter_options = "".join(f'<option value="{escape(n)}">{escape(n)}</option>' for n in fn_names)

    return f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Validazione Serverledge — {escape(scenario_path.name)}</title>
  <style>
    :root {{
      --bg: #f8fafc;
      --surface: #ffffff;
      --surface-alt: #f1f5f9;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --sim: #2563eb;
      --real: #f97316;
      --good: #059669;
      --warn: #d97706;
      --bad: #dc2626;
      --radius: 12px;
      --shadow: 0 1px 3px rgba(15, 23, 42, 0.06), 0 1px 2px rgba(15, 23, 42, 0.04);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
    }}
    .topbar-inner {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 12px 24px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      font-weight: 700;
      font-size: 1rem;
    }}
    .brand-dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--sim), var(--real));
    }}
    .nav {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .nav a {{
      text-decoration: none;
      color: var(--muted);
      font-size: 0.875rem;
      font-weight: 500;
      padding: 6px 12px;
      border-radius: 8px;
      transition: background 0.15s, color 0.15s;
    }}
    .nav a:hover {{
      background: var(--surface-alt);
      color: var(--text);
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 28px;
    }}
    section {{
      scroll-margin-top: 72px;
    }}
    .hero {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 20px 24px;
      box-shadow: var(--shadow);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    .hero h1 {{
      margin: 0 0 6px;
      font-size: 1.45rem;
      letter-spacing: -0.02em;
    }}
    .meta-chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    .chip {{
      background: var(--surface-alt);
      border: 1px solid var(--border);
      padding: 4px 10px;
      border-radius: 999px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 14px;
    }}
    .kpi-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }}
    .kpi-title {{
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .kpi-pair {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 8px;
      margin-bottom: 6px;
    }}
    .kpi-val {{
      font-size: 1.25rem;
      font-weight: 700;
      font-variant-numeric: tabular-nums;
    }}
    .kpi-val.sim {{ color: var(--sim); }}
    .kpi-val.real {{ color: var(--real); }}
    .kpi-sub {{
      font-size: 0.8rem;
      color: var(--muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 1.15rem;
      letter-spacing: -0.01em;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      box-shadow: var(--shadow);
    }}
    .card h3 {{
      margin: 0 0 4px;
      font-size: 0.98rem;
    }}
    .card-desc {{
      margin: 0 0 12px;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    .chart-svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .svg-grid {{ stroke: #e2e8f0; stroke-width: 1; }}
    .svg-axis {{ fill: #64748b; font-size: 11px; }}
    .svg-label {{ fill: #334155; font-size: 11.5px; font-weight: 500; }}
    .svg-label-bold {{ fill: #0f172a; font-size: 12px; font-weight: 600; }}
    .svg-muted {{ fill: #64748b; font-size: 11.5px; }}
    .svg-legend {{ fill: #334155; font-size: 11.5px; font-weight: 500; }}
    .svg-val {{ fill: #334155; font-size: 10px; font-weight: 600; }}
    .svg-na {{ fill: #94a3b8; font-size: 10px; }}
    .svg-inbar {{ fill: #ffffff; font-size: 11.5px; font-weight: 700; }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: 10px;
      border: 1px solid var(--border);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
      background: var(--surface);
    }}
    th, td {{
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
    }}
    th {{
      background: var(--surface-alt);
      font-weight: 600;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.03em;
      color: var(--muted);
    }}
    td.num, th.num {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    tr:last-child td {{ border-bottom: none; }}
    tr.fn-group-start td {{ border-top: 2px solid var(--border); }}
    td.fn-name {{
      font-weight: 700;
      background: #fafcff;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 600;
    }}
    .pill-sim {{ background: #dbeafe; color: #1d4ed8; }}
    .pill-real {{ background: #ffedd5; color: #c2410c; }}
    .pill-completed {{ background: #d1fae5; color: #047857; }}
    .pill-rejected {{ background: #fee2e2; color: #b91c1c; }}
    .pill-pending {{ background: #fef3c7; color: #b45309; }}
    .pill-error {{ background: #e0e7ff; color: #4338ca; }}
    .pill-warm {{ background: #ecfdf5; color: #047857; }}
    .pill-cold {{ background: #eff6ff; color: #1d4ed8; }}
    .pill-region {{ opacity: 0.75; font-weight: 500; }}
    .pill-list {{ display: flex; flex-wrap: wrap; gap: 6px; }}
    .delta {{
      font-size: 0.8rem;
      font-weight: 600;
      padding: 2px 7px;
      border-radius: 6px;
    }}
    .delta.good {{ background: #d1fae5; color: #065f46; }}
    .delta.warn {{ background: #fef3c7; color: #92400e; }}
    .delta.neutral {{ background: var(--surface-alt); color: #334155; }}
    .delta.info {{ background: #e0f2fe; color: #0369a1; }}
    .delta.muted, .muted {{ color: var(--muted); }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 14px;
    }}
    .filters input, .filters select {{
      padding: 7px 11px;
      border: 1px solid var(--border);
      border-radius: 8px;
      font-size: 0.86rem;
      background: var(--surface);
      color: var(--text);
    }}
    .filters input {{ min-width: 220px; }}
    .filter-count {{
      margin-left: auto;
      font-size: 0.82rem;
      color: var(--muted);
    }}
    .empty-msg {{ color: var(--muted); font-size: 0.9rem; }}
    footer {{
      text-align: center;
      font-size: 0.8rem;
      color: var(--muted);
      padding: 12px 0 28px;
    }}
    @media (max-width: 640px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
      main {{ padding: 14px; }}
    }}
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <span class="brand-dot"></span>
        <span>Serverledge Validation Report</span>
      </div>
      <nav class="nav">
        <a href="#overview">Panoramica</a>
        <a href="#charts">Grafici</a>
        <a href="#summary">Riepilogo</a>
        <a href="#functions">Per funzione</a>
        <a href="#placement">Placement &amp; Forward</a>
        <a href="#requests">Richieste reali</a>
      </nav>
    </div>
  </header>

  <main>
    <section class="hero" id="overview">
      <div>
        <h1>Confronto Simulatore vs Serverledge Reale</h1>
        <div class="meta-chips">
          <span class="chip">Scenario: <strong>{escape(scenario_path.name)}</strong></span>
          <span class="chip">Seed: <strong>{seed}</strong></span>
          <span class="chip">Arrivi: <strong>{len(arrivals)}</strong></span>
          <span class="chip">Cutoff: <strong>t = {cutoff_s:.3f} s</strong></span>
          <span class="chip">Esito finale reale: <strong>{real_final_ok}/{len(rows)} OK</strong></span>
        </div>
      </div>
      <div class="meta-chips">
        <span class="chip">Generato: {generated_at}</span>
      </div>
    </section>

    <section class="kpi-grid" aria-label="Indicatori principali">
      <div class="kpi-card">
        <div class="kpi-title">Completate (entro cutoff)</div>
        <div class="kpi-pair">
          <span class="kpi-val sim">{sim_total.completed}</span>
          <span class="kpi-val real">{real_total.completed}</span>
        </div>
        <div class="kpi-sub"><span>Sim vs Reale</span>{_fmt_delta(sim_total.completed, real_total.completed, 'n')}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Rifiutate</div>
        <div class="kpi-pair">
          <span class="kpi-val sim">{sim_total.rejected}</span>
          <span class="kpi-val real">{real_total.rejected}</span>
        </div>
        <div class="kpi-sub"><span>Sim vs Reale</span>{_fmt_delta(sim_total.rejected, real_total.rejected, 'n')}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Risposta media</div>
        <div class="kpi-pair">
          <span class="kpi-val sim">{_fmt_sec(sim_total.mean_response_s)}</span>
          <span class="kpi-val real">{_fmt_sec(real_total.mean_response_s)}</span>
        </div>
        <div class="kpi-sub"><span>Sim vs Reale</span>{_fmt_delta(sim_total.mean_response_s, real_total.mean_response_s, 's')}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Latenza p95</div>
        <div class="kpi-pair">
          <span class="kpi-val sim">{_fmt_sec(sim_total.p95_response_s)}</span>
          <span class="kpi-val real">{_fmt_sec(real_total.p95_response_s)}</span>
        </div>
        <div class="kpi-sub"><span>Sim vs Reale</span>{_fmt_delta(sim_total.p95_response_s, real_total.p95_response_s, 's')}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-title">Cold Start %</div>
        <div class="kpi-pair">
          <span class="kpi-val sim">{_fmt_pct(sim_total.cold_pct)}</span>
          <span class="kpi-val real">{_fmt_pct(real_total.cold_pct)}</span>
        </div>
        <div class="kpi-sub"><span>Sim vs Reale</span>{_fmt_delta(sim_total.cold_pct, real_total.cold_pct, '%')}</div>
      </div>
    </section>

    <section id="charts">
      <h2>Confronto grafico</h2>
      <div class="charts-grid">
        <div class="card">
          <h3>Distribuzione esiti complessivi</h3>
          <p class="card-desc">Confronto tra richieste completate, rifiutate, pendenti ed errori al cutoff t = {cutoff_s:.2f} s.</p>
          {chart_outcomes}
        </div>
        <div class="card">
          <h3>Nodi di esecuzione (sistema reale)</h3>
          <p class="card-desc">Distribuzione delle invocazioni completate sui worker reali Serverledge.</p>
          {chart_nodes}
        </div>
        <div class="card">
          <h3>Tempo di risposta medio per funzione (s)</h3>
          <p class="card-desc">Include attesa in coda, eventuale cold start, offload e tempo di servizio.</p>
          {chart_resp}
        </div>
        <div class="card">
          <h3>Latenza p95 per funzione (s)</h3>
          <p class="card-desc">95° percentile del tempo di risposta sulle richieste completate entro il cutoff.</p>
          {chart_p95}
        </div>
        <div class="card">
          <h3>Tasso di Cold Start per funzione (%)</h3>
          <p class="card-desc">Percentuale di invocazioni servite con avvio di un nuovo container.</p>
          {chart_cold}
        </div>
        <div class="card">
          <h3>Tempo di servizio medio per funzione (s)</h3>
          <p class="card-desc">Verifica l'allineamento tra durate campionate dal simulatore e sleep reali.</p>
          {chart_service}
        </div>
      </div>
    </section>

    <section id="summary" class="card">
      <h2>Riepilogo generale</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Sistema</th>
              <th class="num">Arrivi</th>
              <th class="num">Completate</th>
              <th class="num">Rifiutate</th>
              <th class="num">Pendenti</th>
              <th class="num">Errori</th>
              <th class="num">Cold %</th>
              <th class="num">Risposta media</th>
              <th class="num">p95</th>
              <th class="num">Servizio medio</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><span class="pill pill-sim">Simulatore</span></td>
              <td class="num">{sim_total.arrivals}</td>
              <td class="num">{sim_total.completed}</td>
              <td class="num">{sim_total.rejected}</td>
              <td class="num">{sim_total.pending}</td>
              <td class="num">{sim_total.errors}</td>
              <td class="num">{_fmt_pct(sim_total.cold_pct)}</td>
              <td class="num">{_fmt_sec(sim_total.mean_response_s)}</td>
              <td class="num">{_fmt_sec(sim_total.p95_response_s)}</td>
              <td class="num">{_fmt_sec(sim_total.mean_service_s)}</td>
            </tr>
            <tr>
              <td><span class="pill pill-real">Reale</span></td>
              <td class="num">{real_total.arrivals}</td>
              <td class="num">{real_total.completed}</td>
              <td class="num">{real_total.rejected}</td>
              <td class="num">{real_total.pending}</td>
              <td class="num">{real_total.errors}</td>
              <td class="num">{_fmt_pct(real_total.cold_pct)}</td>
              <td class="num">{_fmt_sec(real_total.mean_response_s)}</td>
              <td class="num">{_fmt_sec(real_total.p95_response_s)}</td>
              <td class="num">{_fmt_sec(real_total.mean_service_s)}</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <section id="functions" class="card">
      <h2>Dettaglio per funzione</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Funzione</th>
              <th>Sistema</th>
              <th class="num">Arrivi</th>
              <th class="num">Completate</th>
              <th class="num">Rifiutate</th>
              <th class="num">Pendenti</th>
              <th class="num">Errori</th>
              <th class="num">Cold %</th>
              <th class="num">Risposta media</th>
              <th class="num">p95</th>
              <th class="num">Servizio medio</th>
              <th class="num">Δ Risposta</th>
            </tr>
          </thead>
          <tbody>
            {"".join(fn_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section id="placement" class="card">
      <h2>Placement e forwarding (Simulatore vs Reale)</h2>
      <p class="card-desc">Confronto dei nodi e delle regioni/aree su cui sono state eseguite le invocazioni completate entro il cutoff (evidenzia eventuali scelte diverse di offloading o discovery dinamico).</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Funzione</th>
              <th>Simulatore (nodi / regioni)</th>
              <th>Reale (nodi / aree)</th>
              <th>Esito routing</th>
            </tr>
          </thead>
          <tbody>
            {"".join(placement_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <section id="requests" class="card">
      <h2>Esploratore richieste reali</h2>
      <div class="filters">
        <input id="req-search" type="search" placeholder="Cerca nodo, area, #seq, HTTP..." />
        <select id="req-fn">
          <option value="">Tutte le funzioni</option>
          {fn_filter_options}
        </select>
        <select id="req-status">
          <option value="">Tutti gli esiti</option>
          <option value="completed">Completate</option>
          <option value="rejected">Rifiutate (429)</option>
          <option value="pending">Pendenti</option>
          <option value="error">Errori</option>
        </select>
        <span class="filter-count" id="req-count"></span>
      </div>
      <div class="table-wrap">
        <table id="req-table">
          <thead>
            <tr>
              <th class="num">#</th>
              <th>Funzione</th>
              <th class="num">Arrivo</th>
              <th class="num">Invio</th>
              <th class="num">Servizio</th>
              <th class="num">Risposta</th>
              <th>Start</th>
              <th>Nodo reale</th>
              <th>Area</th>
              <th class="num">HTTP</th>
              <th>Esito (al cutoff)</th>
            </tr>
          </thead>
          <tbody>
            {"".join(req_rows_html)}
          </tbody>
        </table>
      </div>
    </section>

    <footer>
      File sorgenti: <code>{escape(str(scenario_path))}</code> · <code>{escape(str(results_path))}</code>
    </footer>
  </main>

  <script>
    (function () {{
      const search = document.getElementById('req-search');
      const fnSel = document.getElementById('req-fn');
      const stSel = document.getElementById('req-status');
      const countEl = document.getElementById('req-count');
      const rows = Array.from(document.querySelectorAll('#req-table tbody tr'));

      function applyFilters() {{
        const q = search.value.trim().toLowerCase();
        const fn = fnSel.value;
        const st = stSel.value;
        let visible = 0;
        for (const row of rows) {{
          const matchFn = !fn || row.dataset.fn === fn;
          const matchSt = !st || row.dataset.status === st;
          const matchQ = !q || row.textContent.toLowerCase().includes(q);
          const show = matchFn && matchSt && matchQ;
          row.style.display = show ? '' : 'none';
          if (show) visible++;
        }}
        countEl.textContent = visible + ' / ' + rows.length + ' richieste';
      }}

      search.addEventListener('input', applyFilters);
      fnSel.addEventListener('change', applyFilters);
      stSel.addEventListener('change', applyFilters);
      applyFilters();
    }})();
  </script>
</body>
</html>
"""
