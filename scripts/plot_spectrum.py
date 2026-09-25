#!/usr/bin/env python3
"""Plot a Figure-2-style spectrum from `scan_solar_mhd_spectrum.py` output.

The scan script writes one subdirectory per (m, Le) cell, each
containing `spectrum.npz` (deduped unique modes) and `raw_scan.npz`
(per-target eigenvalues). This tool collects them and renders:

  * **ω vs m** scatter, one series per Le, with γ encoded as marker
    alpha/size — the canonical inertial-mode dispersion diagram.
  * Optionally an **ω vs γ** scatter (the complex-plane view) for a
    fixed m, comparing Le values.

Usage
-----

  python scripts/plot_spectrum.py results/spectrum
  python scripts/plot_spectrum.py results/spectrum --gamma-max 0.5
  python scripts/plot_spectrum.py results/spectrum --complex-plane --m 3
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _collect(spectrum_dir: Path) -> List[Dict]:
    """Load every cell's spectrum.npz under `spectrum_dir`.

    Each cell subdir matches `m{NN}_Le{...}`. Returns a list of dicts
    with keys: m, Le, omega, gamma.
    """
    cells = []
    for sub in sorted(spectrum_dir.iterdir()):
        if not sub.is_dir():
            continue
        spec = sub / "spectrum.npz"
        if not spec.exists():
            continue
        with np.load(spec) as d:
            cells.append({
                "m": int(d["m"]),
                "Le": float(d["Le"]),
                "omega": d["omega"].copy(),
                "gamma": d["gamma"].copy(),
                "Ek": float(d["Ek"]) if "Ek" in d.files else None,
            })
    return cells


def _plot_dispersion(
    cells: List[Dict], out_path: Path,
    gamma_max: float, omega_range: Tuple[float, float],
) -> None:
    """ω vs m scatter, one color per Le, alpha by damping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Le_values = sorted({c["Le"] for c in cells})
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(Le_values)))
    Le_color = {Le: cmap[i] for i, Le in enumerate(Le_values)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in cells:
        keep = (c["gamma"] > -gamma_max) & \
               (c["omega"] >= omega_range[0]) & (c["omega"] <= omega_range[1])
        om = c["omega"][keep]
        ga = c["gamma"][keep]
        if len(om) == 0:
            continue
        # Alpha encodes damping: weakly-damped modes (γ near 0) are
        # opaque, strongly-damped modes faint.
        alpha = np.clip(1.0 + ga / gamma_max, 0.15, 1.0)
        ax.scatter(
            np.full_like(om, c["m"]), om,
            s=28, color=Le_color[c["Le"]],
            alpha=alpha, edgecolors="none",
        )

    # One legend handle per Le.
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=Le_color[Le], markersize=8,
               label=f"Le = {Le:g}")
        for Le in Le_values
    ]
    ax.legend(handles=handles, loc="best", title="Lehnert number")

    ax.set_xlabel("m (azimuthal order)")
    ax.set_ylabel(r"$\omega / \Omega_0$")
    ax.set_title("Inertial-mode spectrum (marker opacity ∝ weak damping)")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5)

    # Mark the n=0 sectoral Rossby asymptote -2/(m+1).
    ms = sorted({c["m"] for c in cells})
    if ms:
        m_arr = np.array(ms)
        ax.plot(m_arr, -2.0/(m_arr+1), "k--", alpha=0.4,
                label="-2/(m+1)", zorder=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_complex_plane(
    cells: List[Dict], out_path: Path, m_target: int,
    gamma_max: float, omega_range: Tuple[float, float],
) -> None:
    """ω vs γ scatter at fixed m, comparing Le values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sel = [c for c in cells if c["m"] == m_target]
    if not sel:
        print(f"No cells with m={m_target}")
        return

    Le_values = sorted({c["Le"] for c in sel})
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(Le_values)))
    Le_color = {Le: cmap[i] for i, Le in enumerate(Le_values)}

    fig, ax = plt.subplots(figsize=(8, 6))
    for c in sel:
        keep = (c["gamma"] > -gamma_max) & \
               (c["omega"] >= omega_range[0]) & (c["omega"] <= omega_range[1])
        ax.scatter(
            c["omega"][keep], c["gamma"][keep],
            s=40, color=Le_color[c["Le"]],
            alpha=0.8, edgecolors="k", linewidths=0.4,
            label=f"Le = {c['Le']:g}",
        )
    ax.set_xlabel(r"$\omega / \Omega_0$")
    ax.set_ylabel(r"$\gamma$ (decay rate)")
    ax.set_title(f"Complex spectrum at m={m_target}")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="r", lw=0.5, alpha=0.5)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot a Figure-2-style spectrum from scan output.",
    )
    parser.add_argument("spectrum_dir", type=Path)
    parser.add_argument("--gamma-max", type=float, default=1.0,
                        help="Hide modes with γ < -gamma_max (too damped).")
    parser.add_argument("--omega-min", type=float, default=-2.0)
    parser.add_argument("--omega-max", type=float, default=2.0)
    parser.add_argument("--complex-plane", action="store_true",
                        help="Also produce an ω–γ scatter at --m.")
    parser.add_argument("--m", type=int, default=3,
                        help="m for the complex-plane plot.")
    args = parser.parse_args()

    if not args.spectrum_dir.is_dir():
        print(f"Not a directory: {args.spectrum_dir}")
        return 1

    cells = _collect(args.spectrum_dir)
    if not cells:
        print(f"No spectrum.npz files found under {args.spectrum_dir}")
        return 1

    print(f"Loaded {len(cells)} cells: "
          f"m={sorted({c['m'] for c in cells})}, "
          f"Le={sorted({c['Le'] for c in cells})}")

    disp_path = args.spectrum_dir / "dispersion.png"
    _plot_dispersion(
        cells, disp_path, args.gamma_max,
        (args.omega_min, args.omega_max),
    )
    print(f"Wrote {disp_path}")

    if args.complex_plane:
        cp_path = args.spectrum_dir / f"complex_m{args.m:02d}.png"
        _plot_complex_plane(
            cells, cp_path, args.m, args.gamma_max,
            (args.omega_min, args.omega_max),
        )
        print(f"Wrote {cp_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
