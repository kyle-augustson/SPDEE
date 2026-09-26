#!/usr/bin/env python3
"""Cross-check the collocated and staggered solar-eigenproblem Figure-2 spectra.

Reads the two .npz files written by polytrope_spectrum.py (one per
core) and reports side by side:

  - the Rossby anchor (both at −2/(m+1));
  - the convective ridge: the fastest-growing clean mode;
  - the largest-Re unstable mode (the red-dot selection of Bekki et al. 2022, ≈ 0.27);
  - the p-mode placement (|Re ω| ~ 1e4 acoustic band);
  - an overlay scatter plot (fig2_compare.png).

Usage (after running both cores):
    python3 scripts/compare_cores_spectrum.py fig2_coloc.npz fig2_stag.npz
    python3 scripts/compare_cores_spectrum.py fig2_coloc.npz fig2_stag.npz \\
        --out fig2_compare.png
"""
from __future__ import annotations

import argparse

import numpy as np


def load(path):
    d = np.load(path)
    om = np.asarray(d["omega"])
    om = om[np.isfinite(om)]
    om = om[np.abs(om) < 1e8]
    return {
        "omega": om,
        "m": int(d["m"]) if "m" in d else None,
        "Ek": float(d["Ek"]) if "Ek" in d else None,
        "backend": str(d["backend"]) if "backend" in d else path,
        "N_r": int(d["N_r"]) if "N_r" in d else None,
        "N_theta": int(d["N_theta"]) if "N_theta" in d else None,
    }


def report(tag, S):
    om = S["omega"]
    m = S["m"]
    anchor = -2.0 / (m + 1) if m is not None else None

    # Rossby anchor: nearest mode to the analytic value
    if anchor is not None:
        near = om[np.argmin(np.abs(om - (anchor + 0j)))]
    else:
        near = None

    # growing modes in the physical inertial band
    g = om[(om.imag > 1e-4) & (np.abs(om.real) < 2.0)]
    fastest = g[np.argmax(g.imag)] if len(g) else None
    # largest-Re unstable mode (red-dot criterion of Bekki et al. 2022); restrict to the
    # convective quadrant Re>0, Im>0 to avoid grabbing a high-Re damped tail
    gq = g[g.real > 0]
    largest_re = gq[np.argmax(gq.real)] if len(gq) else None

    # acoustic p-modes: |Re ω| ~ 1e4
    pmode = om[np.abs(om.real) > 1e3]
    pmax = np.abs(pmode.real).max() if len(pmode) else 0.0

    print(f"\n=== {tag}  ({S['backend']}, {S['N_r']}x{S['N_theta']}, "
          f"m={m}, Ek={S['Ek']:.3e}) ===")
    print(f"  finite modes: {len(om)}")
    if near is not None:
        print(f"  Rossby anchor (−2/{m+1}={anchor:+.4f}): "
              f"nearest {near.real:+.4f}{near.imag:+.4f}j "
              f"(Re err {abs(near.real-anchor):.4f})")
    print(f"  growing in band |Re|<2: {len(g)}")
    if fastest is not None:
        print(f"  FASTEST-growing clean mode: "
              f"{fastest.real:+.4f}{fastest.imag:+.4f}j   "
              f"<- expect Re≈0 (the convective ridge peak)")
    if largest_re is not None:
        print(f"  LARGEST-Re unstable mode:   "
              f"{largest_re.real:+.4f}{largest_re.imag:+.4f}j   "
              f"<- red-dot criterion of Bekki et al. 2022 (theirs ≈ +0.27)")
    if pmax > 0:
        print(f"  acoustic p-modes present up to |Re ω| ≈ {pmax:.0f} "
              f"(compressible, ~1e4 expected)")
    return {"fastest": fastest, "largest_re": largest_re, "anchor": near}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("coloc_npz")
    p.add_argument("stag_npz")
    p.add_argument("--out", default="fig2_compare.png")
    args = p.parse_args()

    C = load(args.coloc_npz)
    Sg = load(args.stag_npz)

    rc = report("COLLOCATED", C)
    rs = report("STAGGERED", Sg)

    print("\n================ CROSS-CHECK VERDICT ================")
    fc, fs = rc["fastest"], rs["fastest"]
    if fc is not None and fs is not None:
        print(f"  Fastest-growing Re:  coloc {fc.real:+.3f}   "
              f"stag {fs.real:+.3f}")
        if abs(fc.real) < 0.1 and abs(fs.real) < 0.1:
            print("  -> BOTH backends peak near Re≈0. The Re≈0 convective peak "
                  "is PHYSICAL, not a collocated-grid artifact.")
        else:
            print("  -> backends DISAGREE on the peak location — investigate "
                  "(grid resolution or a real scheme difference).")
    lc, ls = rc["largest_re"], rs["largest_re"]
    if lc is not None and ls is not None:
        print(f"  Largest-Re unstable: coloc {lc.real:+.3f}   "
              f"stag {ls.real:+.3f}   (red dot of Bekki et al. 2022 ≈ +0.27)")
        print("  -> compare these (converged) to the 0.27 of Bekki et al. 2022; this is the "
              "apples-to-apples red-dot test.")
    print("=====================================================")

    # overlay scatter
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 7))
        for S, color, lab in ((C, "C0", "collocated"),
                              (Sg, "C1", "staggered")):
            om = S["omega"]
            band = om[(np.abs(om.real) < 2.5) & (om.imag > -2) & (om.imag < 1.2)]
            ax.scatter(band.real, band.imag, s=6, alpha=0.4,
                       color=color, label=lab)
        ax.scatter([0.27], [0.36], s=80, color="red", zorder=5,
                   label="red dot, Bekki et al. 2022")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(0, color="k", lw=0.5, ls="--")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2, 1.2)
        ax.set_xlabel(r"$\Re[\omega]/\Omega_0$")
        ax.set_ylabel(r"$\Im[\omega]/\Omega_0$")
        ax.set_title(f"Solar-eigenproblem Fig-2 cross-check  m={C['m']} Ek={C['Ek']:.2e}")
        ax.legend(loc="lower right")
        fig.savefig(args.out, dpi=130, bbox_inches="tight")
        print(f"\n[render] {args.out}")
    except Exception as e:
        print(f"\n[render skipped: {e}]")


if __name__ == "__main__":
    main()
