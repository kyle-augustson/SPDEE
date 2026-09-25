#!/usr/bin/env python3
"""Analyze a directory of sweep .npz files.

Auto-detects the sweep type (ball-by-m or Le-by-amplitude) by inspecting
the schema, then produces:

  * a tab-separated table on stdout
  * a 2-panel PNG plot
  * a markdown summary file in the sweep directory

Designed to be run on the output of either ``scripts/sweep_solar_ball.py``
or ``scripts/sweep_solar_mhd_Le.py`` without any sweep-specific flags.

Usage
-----

  python scripts/analyze_sweep.py path/to/sweep_dir

Outputs go to ``sweep_dir/analysis_summary.md`` and ``sweep_dir/summary.png``.

Optional flags:

  --no-plot              skip the PNG (text-only environments)
  --output-stem NAME     use NAME instead of "summary" / "analysis_summary"
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_all(sweep_dir: Path) -> List[Dict]:
    """Load every .npz in `sweep_dir` into a list of dicts.

    Scalar arrays (ndim == 0) are unwrapped via ``.item()`` so downstream
    code doesn't have to do it. Array fields stay as arrays.
    """
    out = []
    for f in sorted(sweep_dir.glob("*.npz")):
        with np.load(f) as d:
            rec = {}
            for k in d.files:
                v = d[k]
                rec[k] = v.item() if v.ndim == 0 else v.copy()
            rec["__file"] = f.name
            out.append(rec)
    return out


def _detect_sweep_kind(records: List[Dict]) -> str:
    """Return 'le' for an Le-sweep, 'ball' for a ball-by-m sweep.

    The discriminator is the presence of the ``Le`` key in the npz
    schema. The Le-sweep script saves it; the ball-sweep script
    does not.
    """
    if not records:
        return "empty"
    if "Le" in records[0]:
        return "le"
    return "ball"


# ---------------------------------------------------------------------------
# Ball sweep analysis
# ---------------------------------------------------------------------------

def _analyze_ball(records: List[Dict]) -> Dict:
    """Build a summary structure for a ball-by-m sweep.

    Records have keys: m, N_r, N_theta, omega, gamma, t_lu_decompose,
    t_arpack, ...
    """
    cells = sorted({(int(r["N_r"]), int(r["N_theta"])) for r in records})
    m_values = sorted({int(r["m"]) for r in records})

    # Build a lookup: by_cell[(N_r, N_θ)][m] = record.
    by_cell: Dict[Tuple[int, int], Dict[int, Dict]] = {}
    for r in records:
        cell = (int(r["N_r"]), int(r["N_theta"]))
        m = int(r["m"])
        by_cell.setdefault(cell, {})[m] = r

    # Headline table: for each (cell, m), the closest-to-target ω, γ.
    rows = []
    for m in m_values:
        row = {"m": m, "target": -2.0/(m+1)}
        for cell in cells:
            rec = by_cell.get(cell, {}).get(m)
            if rec is None:
                row[f"ω_{cell}"] = float("nan")
                row[f"γ_{cell}"] = float("nan")
            else:
                # Index 0 is closest-to-target after the sweep's own sort.
                row[f"ω_{cell}"] = float(rec["omega"][0])
                row[f"γ_{cell}"] = float(rec["gamma"][0])
        rows.append(row)

    # Convergence diagnostics: for every PAIR of cells, compute |Δω|
    # and |Δγ| per m. The threshold 1e-3 separates "smooth convergence"
    # from "mode jump" — the latter signals a dense spectrum where mode
    # tracking by eigenvalue proximity fails.
    #
    # We collect diagnostics for every cell pair, then label each
    # axis of comparison (N_r-doubling, N_θ-doubling, both) so the
    # reader can see at a glance which axis the mode jumps live on.
    conv_diagnostics = []
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            cell_a, cell_b = cells[i], cells[j]
            # Label the comparison axis.
            if cell_a[0] != cell_b[0] and cell_a[1] == cell_b[1]:
                axis = "N_r"
            elif cell_a[0] == cell_b[0] and cell_a[1] != cell_b[1]:
                axis = "N_θ"
            else:
                axis = "both"
            per_m_diff = []
            for m in m_values:
                ra = by_cell.get(cell_a, {}).get(m)
                rb = by_cell.get(cell_b, {}).get(m)
                if ra is None or rb is None:
                    continue
                d_om = abs(float(ra["omega"][0]) - float(rb["omega"][0]))
                d_ga = abs(float(ra["gamma"][0]) - float(rb["gamma"][0]))
                per_m_diff.append({
                    "m": m, "d_omega": d_om, "d_gamma": d_ga,
                    "mode_jump": d_om > 1e-3,
                })
            conv_diagnostics.append({
                "pair": (cell_a, cell_b),
                "axis": axis,
                "per_m": per_m_diff,
            })

    # Timing roll-up per cell.
    timing = []
    for cell in cells:
        ts = [by_cell[cell][m] for m in by_cell[cell]]
        timing.append({
            "cell": cell,
            "n_solves": len(ts),
            "lu_avg": float(np.mean([t["t_lu_decompose"] for t in ts])),
            "arpack_avg": float(np.mean([t["t_arpack"] for t in ts])),
            "solve_avg": float(np.mean([t["solve_time"] for t in ts])),
            "rss_max_avg": float(np.mean([t["rss_peak_kb"] for t in ts])) / 1024 / 1024,
        })

    return {
        "kind": "ball",
        "cells": cells,
        "m_values": m_values,
        "rows": rows,
        "convergence": conv_diagnostics,
        "timing": timing,
        "by_cell": by_cell,
    }


def _format_ball_text(s: Dict) -> str:
    """Plain-text rendering of the ball-sweep summary."""
    lines = []
    lines.append(f"Ball sweep: {len(s['cells'])} cells × {len(s['m_values'])} m values")
    lines.append(f"  cells: {s['cells']}")
    lines.append(f"  m: {s['m_values']}")
    lines.append("")

    # ω table
    lines.append("ω (closest to target -2/(m+1)):")
    header = f"  {'m':>3}  {'target':>9}  "
    for cell in s["cells"]:
        header += f"{f'ω({cell[0]}×{cell[1]})':>16}  "
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row in s["rows"]:
        line = f"  {row['m']:>3}  {row['target']:>+9.5f}  "
        for cell in s["cells"]:
            v = row[f"ω_{cell}"]
            line += f"{v:>+16.7f}  " if not np.isnan(v) else f"{'(missing)':>16}  "
        lines.append(line)

    # γ table
    lines.append("")
    lines.append("γ (decay rates):")
    header = f"  {'m':>3}  {'':>9}  "
    for cell in s["cells"]:
        header += f"{f'γ({cell[0]}×{cell[1]})':>16}  "
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for row in s["rows"]:
        line = f"  {row['m']:>3}  {'':>9}  "
        for cell in s["cells"]:
            v = row[f"γ_{cell}"]
            line += f"{v:>+16.7f}  " if not np.isnan(v) else f"{'(missing)':>16}  "
        lines.append(line)

    # Convergence diagnostics
    if s["convergence"]:
        for cmp_entry in s["convergence"]:
            lines.append("")
            cell_a, cell_b = cmp_entry["pair"]
            lines.append(
                f"Convergence {cell_a[0]}×{cell_a[1]} → "
                f"{cell_b[0]}×{cell_b[1]} ({cmp_entry['axis']} axis):"
            )
            lines.append(f"  {'m':>3}  {'|Δω|':>11}  {'|Δγ|':>11}  flag")
            for d in cmp_entry["per_m"]:
                flag = "  ← mode jump" if d["mode_jump"] else ""
                lines.append(
                    f"  {d['m']:>3}  {d['d_omega']:>11.2e}  "
                    f"{d['d_gamma']:>11.2e}{flag}"
                )

    # Timing
    lines.append("")
    lines.append("Timing (avg per cell):")
    lines.append(f"  {'cell':>12}  {'LU':>8}  {'ARPACK':>8}  {'solve':>8}  {'RSS max':>9}")
    for t in s["timing"]:
        lines.append(
            f"  {t['cell'][0]}×{t['cell'][1]:<5d}  "
            f"{t['lu_avg']:>7.1f}s  {t['arpack_avg']:>7.1f}s  "
            f"{t['solve_avg']:>7.1f}s  {t['rss_max_avg']:>6.2f} GB"
        )

    return "\n".join(lines)


def _format_ball_markdown(s: Dict, sweep_dir: Path) -> str:
    """Markdown rendering."""
    lines = []
    lines.append(f"# Ball sweep analysis — {sweep_dir.name}")
    lines.append("")
    lines.append(f"**{len(s['cells'])} cells × {len(s['m_values'])} m values = "
                 f"{sum(len(s['by_cell'][c]) for c in s['cells'])} solves.**")
    lines.append("")
    lines.append(f"- Cells (N_r × N_θ): `{s['cells']}`")
    lines.append(f"- m values: `{s['m_values']}`")
    lines.append("")

    # ω table as markdown.
    lines.append("## ω (closest-to-target eigenvalue)")
    lines.append("")
    h = "| m | target |"
    sep = "|---|---|"
    for cell in s["cells"]:
        h += f" ω({cell[0]}×{cell[1]}) |"
        sep += "---|"
    lines.append(h)
    lines.append(sep)
    for row in s["rows"]:
        r = f"| {row['m']} | {row['target']:+.5f} |"
        for cell in s["cells"]:
            v = row[f"ω_{cell}"]
            r += f" {v:+.7f} |" if not np.isnan(v) else " (missing) |"
        lines.append(r)

    if s["convergence"]:
        lines.append("")
        lines.append("## Convergence diagnostics")
        lines.append("")
        lines.append("A mode-jump flag indicates `|Δω| > 10⁻³`, which signals "
                     "that the closest-to-target eigenvalue is *not the same physical mode* "
                     "at the two resolutions — typical in dense-spectrum regions where the "
                     "sectoral Rossby sits among magneto-inertial overtones.")
        for cmp_entry in s["convergence"]:
            cell_a, cell_b = cmp_entry["pair"]
            lines.append("")
            lines.append(
                f"### {cell_a[0]}×{cell_a[1]} → {cell_b[0]}×{cell_b[1]} "
                f"({cmp_entry['axis']} axis)"
            )
            lines.append("")
            lines.append("| m | \\|Δω\\| | \\|Δγ\\| | mode jump? |")
            lines.append("|---|---|---|---|")
            for d in cmp_entry["per_m"]:
                flag = "**yes**" if d["mode_jump"] else "no"
                lines.append(
                    f"| {d['m']} | {d['d_omega']:.2e} | {d['d_gamma']:.2e} | {flag} |"
                )

    lines.append("")
    lines.append("## Timing summary")
    lines.append("")
    lines.append("| cell | LU avg | ARPACK avg | solve avg | RSS max avg |")
    lines.append("|---|---|---|---|---|")
    for t in s["timing"]:
        lines.append(
            f"| {t['cell'][0]}×{t['cell'][1]} | "
            f"{t['lu_avg']:.1f}s | {t['arpack_avg']:.1f}s | "
            f"{t['solve_avg']:.1f}s | {t['rss_max_avg']:.2f} GB |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Le sweep analysis
# ---------------------------------------------------------------------------

def _analyze_le(records: List[Dict]) -> Dict:
    """Build a summary structure for a Le-sweep."""
    m_values = sorted({int(r["m"]) for r in records})
    Le_values = sorted({float(r["Le"]) for r in records})

    by_m: Dict[int, Dict[float, Dict]] = {}
    for r in records:
        by_m.setdefault(int(r["m"]), {})[float(r["Le"])] = r

    # For each m, compute Δω/Le² and Δγ/Le² relative to Le=0 (if present).
    # These ratios should be roughly constant for small Le — the
    # perturbative scaling law.
    per_m = []
    for m in m_values:
        Le_recs = by_m[m]
        Le_sorted = sorted(Le_recs.keys())
        # Anchor at Le=0 if present, else at the smallest Le.
        anchor = 0.0 if 0.0 in Le_recs else Le_sorted[0]
        om0 = float(Le_recs[anchor]["rossby_omega"])
        ga0 = float(Le_recs[anchor]["rossby_gamma"])
        scaling = []
        for Le in Le_sorted:
            rec = Le_recs[Le]
            om = float(rec["rossby_omega"])
            ga = float(rec["rossby_gamma"])
            d_om = om - om0
            d_ga = ga - ga0
            scaling.append({
                "Le": Le,
                "omega": om, "gamma": ga,
                "d_omega": d_om, "d_gamma": d_ga,
                "dom_over_le2": (d_om / Le**2) if Le > 0 else float("nan"),
                "dga_over_le2": (d_ga / Le**2) if Le > 0 else float("nan"),
            })
        per_m.append({"m": m, "anchor": anchor, "scaling": scaling})

    # Timing roll-up.
    timing = {
        "n_solves": len(records),
        "lu_avg": float(np.mean([r["t_lu_decompose"] for r in records])),
        "arpack_avg": float(np.mean([r["t_arpack"] for r in records])),
        "solve_avg": float(np.mean([r["solve_time"] for r in records])),
        "rss_max_avg": float(np.mean([r["rss_peak_kb"] for r in records])) / 1024 / 1024,
    }

    # Pull the grid + physics params from the first record (uniform per
    # sweep).
    head = records[0]
    params = {
        "Ek": float(head["Ek"]),
        "Pm": float(head["Pm"]),
        "amplitude_ratio": float(head["amplitude_ratio"]),
        "N_r": int(head["N_r"]), "N_theta": int(head["N_theta"]),
        "r_inner": float(head["r_inner"]), "r_outer": float(head["r_outer"]),
    }

    return {
        "kind": "le",
        "m_values": m_values,
        "Le_values": Le_values,
        "per_m": per_m,
        "timing": timing,
        "params": params,
        "by_m": by_m,
    }


def _format_le_text(s: Dict) -> str:
    lines = []
    p = s["params"]
    lines.append(f"Le sweep: {len(s['m_values'])} m × {len(s['Le_values'])} Le values "
                 f"= {p['N_r']}×{p['N_theta']}, Ek={p['Ek']:.1e}, Pm={p['Pm']:.1f}, "
                 f"|B(r_in)/B(r_out)|={p['amplitude_ratio']:.0e}")
    lines.append("")

    for entry in s["per_m"]:
        m = entry["m"]
        lines.append(f"=== m = {m} ===")
        lines.append(f"  anchor: Le={entry['anchor']}")
        lines.append(f"  {'Le':>10}  {'ω':>10}  {'γ':>10}  "
                     f"{'Δω':>11}  {'Δγ':>11}  {'|Δω|/Le²':>10}  {'|Δγ|/Le²':>10}")
        for row in entry["scaling"]:
            r2 = f"{abs(row['dom_over_le2']):>10.4f}" if not np.isnan(row['dom_over_le2']) else f"{'—':>10}"
            r3 = f"{abs(row['dga_over_le2']):>10.4f}" if not np.isnan(row['dga_over_le2']) else f"{'—':>10}"
            lines.append(
                f"  {row['Le']:>10.2e}  {row['omega']:>+10.5f}  {row['gamma']:>+10.5f}  "
                f"{row['d_omega']:>+11.2e}  {row['d_gamma']:>+11.2e}  {r2}  {r3}"
            )
        lines.append("")

    t = s["timing"]
    lines.append(f"Timing avg: LU={t['lu_avg']:.1f}s, "
                 f"ARPACK={t['arpack_avg']:.1f}s, "
                 f"solve={t['solve_avg']:.1f}s, "
                 f"RSS={t['rss_max_avg']:.2f} GB")
    return "\n".join(lines)


def _format_le_markdown(s: Dict, sweep_dir: Path) -> str:
    lines = []
    p = s["params"]
    lines.append(f"# Le-sweep analysis — {sweep_dir.name}")
    lines.append("")
    lines.append("**Bekki + dipolar B₀ Le-sweep**.")
    lines.append("")
    lines.append(f"- Grid: N_r×N_θ = {p['N_r']}×{p['N_theta']}")
    lines.append(f"- Domain: r ∈ [{p['r_inner']}, {p['r_outer']}]")
    lines.append(f"- Ek = {p['Ek']:.1e}, Pm = {p['Pm']:.1f}, "
                 f"|B(r_in)/B(r_out)| = {p['amplitude_ratio']:.0e}")
    lines.append(f"- m values: {s['m_values']}")
    lines.append(f"- Le values: {s['Le_values']}")
    lines.append(f"- Total solves: {s['timing']['n_solves']}")
    lines.append("")

    for entry in s["per_m"]:
        m = entry["m"]
        lines.append(f"## m = {m}")
        lines.append("")
        lines.append("| Le | ω | γ | Δω | Δγ | \\|Δω\\|/Le² | \\|Δγ\\|/Le² |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in entry["scaling"]:
            r2 = f"{abs(row['dom_over_le2']):.4f}" if not np.isnan(row['dom_over_le2']) else "—"
            r3 = f"{abs(row['dga_over_le2']):.4f}" if not np.isnan(row['dga_over_le2']) else "—"
            lines.append(
                f"| {row['Le']:.2e} | {row['omega']:+.5f} | {row['gamma']:+.5f} | "
                f"{row['d_omega']:+.2e} | {row['d_gamma']:+.2e} | {r2} | {r3} |"
            )
        lines.append("")

        # Verdict on Le² scaling.
        small_le = [r for r in entry["scaling"] if 0 < r["Le"] <= 0.05]
        if len(small_le) >= 2:
            ratios = [abs(r["dom_over_le2"]) for r in small_le]
            spread = (max(ratios) - min(ratios)) / max(max(ratios), 1e-30)
            verdict = "**clean Le² scaling**" if spread < 0.05 else (
                "Le² scaling with %.0f%% spread" % (spread*100)
            )
            lines.append(f"_{verdict} at small Le: |Δω|/Le² ≈ {np.mean(ratios):.4f} ± "
                         f"{(max(ratios)-min(ratios))/2:.4f}_")
            lines.append("")

    lines.append("## Timing summary")
    lines.append("")
    t = s["timing"]
    lines.append(f"- Solves: {t['n_solves']}")
    lines.append(f"- LU avg: {t['lu_avg']:.1f} s")
    lines.append(f"- ARPACK avg: {t['arpack_avg']:.1f} s")
    lines.append(f"- solve total avg: {t['solve_avg']:.1f} s")
    lines.append(f"- RSS peak avg: {t['rss_max_avg']:.2f} GB")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting (optional; skipped via --no-plot)
# ---------------------------------------------------------------------------

def _plot_ball(s: Dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ms = np.array(s["m_values"])
    markers = ["o", "s", "^", "D", "v"]
    for ax, key, ylabel in [
        (axes[0], "ω", r"$\omega$ (closest to target)"),
        (axes[1], "γ", r"$\gamma$ (decay rate)"),
    ]:
        for cell, mk in zip(s["cells"], markers):
            vals = []
            for row in s["rows"]:
                vals.append(row[f"{key}_{cell}"])
            ax.plot(ms, vals, mk + "-", label=f"{cell[0]}×{cell[1]}", alpha=0.85, ms=7)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    # Reference line on ω.
    axes[0].plot(ms, -2.0/(ms+1), "k--", alpha=0.5, label="-2/(m+1) asymptote")
    axes[0].legend(loc="best")
    axes[0].set_title("Ball sweep: closest-to-target mode")

    axes[1].set_xlabel("m")
    axes[1].axhline(0, color="r", lw=0.5, alpha=0.5)
    axes[1].set_xticks(ms)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _plot_le(s: Dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2
    fig, axes = plt.subplots(1, n_panels, figsize=(12, 4.5))

    # Panel 1: ω vs Le and γ vs Le.
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(s["per_m"])))
    for entry, c in zip(s["per_m"], colors):
        Le = np.array([r["Le"] for r in entry["scaling"]])
        om = np.array([r["omega"] for r in entry["scaling"]])
        ax.plot(Le, om, "o-", color=c, label=f"m={entry['m']}", ms=7)
    ax.set_xlabel("Le")
    ax.set_ylabel(r"$\omega$ (Rossby mode)")
    ax.set_xscale("symlog", linthresh=1e-4)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Rossby ω vs Le")

    # Panel 2: |Δω| vs Le on log-log to show Le² slope.
    ax = axes[1]
    for entry, c in zip(s["per_m"], colors):
        scaling = [r for r in entry["scaling"] if r["Le"] > 0]
        Le = np.array([r["Le"] for r in scaling])
        d_om = np.array([abs(r["d_omega"]) for r in scaling])
        ax.loglog(Le, d_om, "o-", color=c, label=f"m={entry['m']}", ms=7)

    # Reference Le² line.
    if s["per_m"] and len(s["per_m"][0]["scaling"]) >= 2:
        Le_ref = np.array([r["Le"] for r in s["per_m"][0]["scaling"] if r["Le"] > 0])
        ax.loglog(Le_ref, 0.17 * Le_ref**2, "k--", alpha=0.5, label="0.17·Le²")

    ax.set_xlabel("Le")
    ax.set_ylabel(r"$|\Delta\omega|$")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_title(r"|Δω| vs Le on log-log (slope-2 indicates Le²)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze a directory of sweep .npz files.",
    )
    parser.add_argument(
        "sweep_dir", type=Path,
        help="Directory containing sweep .npz files.",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip writing the PNG plot.",
    )
    parser.add_argument(
        "--output-stem", default=None,
        help="Stem for output files (default: analysis_summary / summary).",
    )
    args = parser.parse_args()

    if not args.sweep_dir.is_dir():
        print(f"Not a directory: {args.sweep_dir}")
        return 1

    records = _load_all(args.sweep_dir)
    if not records:
        print(f"No .npz files in {args.sweep_dir}")
        return 1

    kind = _detect_sweep_kind(records)
    if kind == "le":
        s = _analyze_le(records)
        text = _format_le_text(s)
        md = _format_le_markdown(s, args.sweep_dir)
        plot_fn = _plot_le
    elif kind == "ball":
        s = _analyze_ball(records)
        text = _format_ball_text(s)
        md = _format_ball_markdown(s, args.sweep_dir)
        plot_fn = _plot_ball
    else:
        print("No recognized sweep type.")
        return 1

    print(text)

    stem = args.output_stem or "analysis_summary"
    md_path = args.sweep_dir / f"{stem}.md"
    md_path.write_text(md)
    print(f"\nWrote markdown: {md_path}")

    if not args.no_plot:
        png_path = args.sweep_dir / "summary.png"
        try:
            plot_fn(s, png_path)
            print(f"Wrote plot: {png_path}")
        except Exception as e:
            print(f"Plot failed: {e!r}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
