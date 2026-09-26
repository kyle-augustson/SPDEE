#!/usr/bin/env python3
"""Classify the growing modes of the solar-eigenproblem Figure-2 system by their
eigenfunctions.

For each growing mode in the inertial band the script pulls the eigenvector
and reports the diagnostics that separate a columnar convective mode from a
spurious one:

  - KE partition (v_r, v_θ, v_φ): a columnar convective mode is v_φ-dominant
    outside the tangent cylinder with substantial v_r (Bekki et al. 2022, §3.2.1); a toroidal
    Rossby-like mode is v_θ-dominant; a spurious scalar mode has little KE
    coherence.
  - v_r↔s₁ phase: convective growth requires Re⟨v_r s₁*⟩ > 0 (buoyancy does
    positive work). Quadrature (~90°) ⇒ no buoyancy work.
  - Nyquist (checkerboard) content of s₁: fraction of s₁ power at the grid
    Nyquist (+−+− cell-to-cell). High ⇒ collocated-grid scalar artifact.
  - radial-node structure of v_φ at the equator (the convective selector of Bekki et al. 2022:
    n_radial(v_r)=0, n_radial(v_φ)=1).

Run (single process; shift-invert per target, fast):
    python3 scripts/eigenfunction_inspect.py
    python3 scripts/eigenfunction_inspect.py --shift-re 0.0 --shift-im 0.9   # Re~0 band
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _Path

import numpy as np

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spdee.backgrounds.polytrope import PolytropeBackground
from spdee.applications.solar_eigenmodes import make_polytrope_system
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem


def nyquist_fraction(field2d):
    """Fraction of |field|² power at the grid-Nyquist (checkerboard) mode."""
    F = np.fft.fft2(field2d)
    P = np.abs(F) ** 2
    tot = P.sum()
    if tot <= 0:
        return 0.0
    Nr, Nth = field2d.shape
    # Nyquist row/col indices
    inr, int_ = Nr // 2, Nth // 2
    # power in the high-wavenumber quadrant (near Nyquist in both directions)
    hi = P[max(inr - 1, 0):inr + 2, max(int_ - 1, 0):int_ + 2].sum()
    return float(hi / tot)


def radial_nodes(field_r):
    """Count sign changes of Re(field) along radius (at equator)."""
    s = np.sign(field_r.real)
    s = s[s != 0]
    return int(np.sum(np.abs(np.diff(s)) > 0))


def classify(ev, om, grid):
    R, T = grid.mesh()
    vol = R**2 * np.sin(T)
    vr = ev["v"][0]
    vth = (ev["v"][1] + ev["v"][-1]) / np.sqrt(2)
    vph = 1j * (ev["v"][-1] - ev["v"][1]) / np.sqrt(2)
    s1 = ev["s1"]

    Er = np.sum(vol * np.abs(vr) ** 2)
    Eth = np.sum(vol * np.abs(vth) ** 2)
    Eph = np.sum(vol * np.abs(vph) ** 2)
    tot = Er + Eth + Eph + 1e-300

    C = (np.sum(vol * vr * np.conj(s1))
         / np.sqrt(np.sum(vol * np.abs(vr) ** 2)
                   * np.sum(vol * np.abs(s1) ** 2) + 1e-300))

    nyq_s1 = nyquist_fraction(s1)
    # equatorial radial cut (nearest theta=pi/2)
    jeq = np.argmin(np.abs(T[0, :] - np.pi / 2))
    n_vr = radial_nodes(vr[:, jeq])
    n_vph = radial_nodes(vph[:, jeq])

    return {
        "om": om,
        "ke": (Er / tot, Eth / tot, Eph / tot),
        "vrs1_phase_deg": float(np.angle(C, deg=True)),
        "vrs1_ReC": float(C.real),
        "nyq_s1": nyq_s1,
        "n_vr": n_vr, "n_vph": n_vph,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Nr", type=int, default=16)
    p.add_argument("--Ntheta", type=int, default=72)
    p.add_argument("--Ek", type=float, default=7.123e-6)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--shift-re", type=float, default=0.27)
    p.add_argument("--shift-im", type=float, default=0.30)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--select", choices=["fastest", "largest-re"],
                   default="fastest",
                   help="which growing mode family to rank first: 'fastest' "
                        "(max Im) or 'largest-re' (red-dot mode of Bekki et al. 2022, Fig. 2)")
    p.add_argument("--order_r", type=int, default=4)
    p.add_argument("--order_theta", type=int, default=4)
    args = p.parse_args()

    poly = PolytropeBackground(delta=args.delta)
    sys_ = make_polytrope_system(poly, m_val=args.m, Ek=args.Ek)
    grid = make_shell_grid(N_r=args.Nr, N_theta=args.Ntheta,
                           r_inner=0.71, r_outer=0.985)
    cp = compile_problem(sys_, grid, m=args.m,
                         order_r=args.order_r, order_theta=args.order_theta)

    sigma = 1j * (args.shift_re + 1j * args.shift_im)
    print(f"[inspect] shift ω={args.shift_re:+.3f}{args.shift_im:+.3f}j, "
          f"k={args.k}, {args.Nr}x{args.Ntheta}, Ek={args.Ek:.2e}", flush=True)
    res = cp.solve_evp(k=args.k, sigma=sigma, return_eigenvectors=True)
    om = -1j * np.asarray(res.eigenvalues)

    # growing modes in the inertial band, sorted by Im desc
    idx = [i for i, w in enumerate(om)
           if np.isfinite(w) and w.imag > 1e-4 and abs(w.real) < 2.0]

    if args.select == "largest-re":
        # The Fig. 2 red-dot criterion of Bekki et al. (2022): the unstable mode with the LARGEST
        # real frequency (the most strongly prograde columnar convective mode,
        # the cleanest archetype — see §3.2, Table 2). Sort by Re desc.
        idx.sort(key=lambda i: -om[i].real)
        print("\n  SELECT=largest-Re unstable mode (red-dot criterion of Bekki et al. 2022).")
    else:
        idx.sort(key=lambda i: -om[i].imag)
        print("\n  SELECT=fastest-growing (largest Im).")

    print(f"  {len(idx)} growing modes in band near this shift. "
          f"Diagnostics (KE=v_r/v_θ/v_φ frac):")
    print("   ω                KE(r/θ/φ)         vr·s1 phase  Re(C)   "
          "Nyq(s1)  n_vr n_vφ")
    for i in idx[:12]:
        d = classify(res.eigenvectors[i], om[i], grid)
        kr, kt, kp = d["ke"]
        w = d["om"]
        print(f"   {w.real:+.3f}{w.imag:+.3f}j   "
              f"{kr:.2f}/{kt:.2f}/{kp:.2f}     "
              f"{d['vrs1_phase_deg']:+6.0f}°   {d['vrs1_ReC']:+.2f}   "
              f"{d['nyq_s1']:.2f}     {d['n_vr']}   {d['n_vph']}")
    print("\n  Columnar convective (Bekki et al. 2022, §3.2.1): v_φ-dominant, substantial "
          "v_r, Re(C)>0 (in-phase), LOW Nyq(s1), n_vr=0, n_vφ=1.")
    print("  Spurious/checkerboard: HIGH Nyq(s1), incoherent KE, Re(C)≈0.")
    print("  Toroidal (Rossby-like): v_θ-dominant, tiny v_r.")


if __name__ == "__main__":
    main()
