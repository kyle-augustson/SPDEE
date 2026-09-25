#!/usr/bin/env python
"""Reproduce Bekki et al. (2022) Figure 2b — the inertial-mode eigenvalue
spectrum in the complex ω-plane — and dump it for cross-correlation.

Two modes:

  (1) SPECTRUM (default): dense full eigenvalue solve (matches Bekki's LAPACK
      approach), maps λ→ω, renders on Fig-2b axes, saves eigenvalues to .npz
      and a scatter .png. Also prints sanity diagnostics (Rossby mode location,
      convective/growing-mode count, high-|Im| artifact fraction).

  (2) EIGENMODE (--eigenmode-omega RE IM): shift-invert solve near a target ω to
      pull a single eigenvector, then save the meridional v_r / v_θ fields for
      eigenfunction comparison against the panels under Fig 2.

Parameters default to Bekki Fig 2: m=8, uniform rotation, δ=10⁻⁶ weakly
superadiabatic, ν=κ=1×10¹¹ cm²/s ⇒ Ek=ν/(Ω₀R_sun²)=7.12×10⁻⁶, stress-free +
impenetrable, Neumann entropy, (N_r,N_θ)=(16,72), CZ shell r∈[0.71,0.985].

The Rossby anchor is Re[ω]/Ω₀ = −2/(m+1) = −2/9 ≈ −0.222 for m=8.

USAGE:

    # full spectrum, collocated:
    python polytrope_spectrum.py --backend collocated --out fig2_coloc

    # the staggered cross-check (the ρ₀-strain term is negligible at
    # Ek = 7.12e-6; --no-strain drops it):
    python polytrope_spectrum.py --backend staggered --out fig2_stag

    # pull one eigenmode near the non-toroidal inertial dot (-0.6, -0.05):
    python polytrope_spectrum.py --eigenmode-omega -0.6 -0.05 --out mode_green
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# Path setup — script lives in scripts/, code lives in spdee/. Lets the
# script run from anywhere without requiring spdee to be pip-installed.
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Bekki Fig-2 nondimensionalization: length R_sun, time 1/Ω₀ (Carrington).
#   Ek = ν / (Ω₀ R_sun²).  Anchor: ν=1e12 cm²/s ⇒ Ek=7.2e-5.
#   IMPORTANT: Bekki 2022 quotes TWO diffusivities. The Fig-2 CAPTION (the m=8
#   overview spectrum reproduced here) uses ν=κ=1e11 cm²/s ⇒ Ek=7.12e-6.
#   The §2.5 / later-figure value is 2e12 ⇒ Ek=1.44e-4 — ~20× larger, which
#   OVER-DAMPS the entire inertial band (every band mode acquires |Im ω|≫0) and
#   suppresses the marginally-growing convective modes. Use the Fig-2 value here.
EK_FIG2 = 7.123e-6   # ν=κ=1e11 cm²/s, Ω₀=2π·456nHz, R_sun=7e10 cm (Fig-2 caption)


def build(args):
    """Build + compile the Bekki Fig-2 system on the chosen backend."""
    from spdee.backgrounds.polytrope import PolytropeBackground
    from spdee.applications.solar_eigenmodes import make_polytrope_system
    from spdee.backends.fd2.grid import make_shell_grid
    from spdee.backends.fd2.compiled_problem import compile_problem

    poly = PolytropeBackground(
        r_inner=args.r_inner, r_outer=args.r_outer, delta=args.delta,
    )
    sys_ = make_polytrope_system(
        poly, m_val=args.m, Ek=args.Ek,
        entropy_bc=args.entropy_bc, geometry="shell",
        with_grad_lnrho_strain=not args.no_strain,
    )
    grid = make_shell_grid(
        N_r=args.N_r, N_theta=args.N_theta,
        r_inner=args.r_inner, r_outer=args.r_outer,
    )
    cp = compile_problem(
        sys_, grid, m=args.m,
        order_r=args.order_r, order_theta=args.order_theta,
        discretization=args.backend,
    )
    return cp, grid


def lam_to_omega(lam):
    """spdee convention λ = +iω ⇒ ω = −iλ. Bekki plots (Re ω, Im ω)/Ω₀.

    Re(ω) = +Im(λ) (frequency), Im(ω) = −Re(λ) (growth; <0 ⇒ damped).
    Canonical derivation lives in
    `spdee.backends.fd2.compiled_problem.EvpResult.eigenvalues_to_omega`;
    this mirrors it so the standalone script needs no spdee import here.
    """
    return -1j * np.asarray(lam)


def run_spectrum(args):
    import scipy.linalg as sla

    t0 = time.time()
    cp, grid = build(args)
    K, M = cp.assemble_evp_matrices()
    n = K.shape[0]
    print(f"[build+assemble] {time.time()-t0:.1f}s   K {K.shape}  nnz {K.nnz}",
          flush=True)

    # Dense generalized eigensolve (all n eigenvalues), matching Bekki's LAPACK.
    # Memory: a single complex128 n×n array is n²·16 B ≈ 0.53 GB at n=5760, but
    # scipy.linalg.eig (LAPACK zggev) densifies BOTH K and M and keeps several
    # n×n scratch/working copies live during the QZ iteration even with
    # right=False, so PEAK RSS is ~4–5× that, i.e. ~2.3 GB observed. The sparse
    # K,M themselves are only a few MB; the cost is entirely the dense solve.
    print(f"[dense eig] solving {n}×{n} generalized EVP "
          f"(slow/heavy step, ~2.3 GB peak at n=5760) ...", flush=True)
    t1 = time.time()
    Kd = np.asarray(K.todense())
    Md = np.asarray(M.todense())
    lam = sla.eig(Kd, Md, right=False)
    print(f"[dense eig] {time.time()-t1:.1f}s", flush=True)

    lam = lam[np.isfinite(lam)]
    omega = lam_to_omega(lam)

    # Drop the obvious constraint/inf-eigenvalue artifacts (|ω| huge). The
    # acoustic p-modes live at |Re ω| ~ 10⁴; Fig 2b is the inertial zoom
    # |Re ω| < ~2.5, |Im ω| < ~2. Keep a generous window for the .npz; the
    # plot uses the Fig-2b axes.
    finite = omega[np.abs(omega) < 1e6]
    np.savez(
        args.out + ".npz",
        omega=finite, lam=lam, m=args.m, Ek=args.Ek, delta=args.delta,
        N_r=args.N_r, N_theta=args.N_theta, backend=args.backend,
    )
    print(f"[save] {args.out}.npz  ({len(finite)} finite eigenvalues)",
          flush=True)

    _diagnostics(finite, args)
    _render(finite, args)


def _diagnostics(omega, args):
    """Cheap sanity checks so you can eyeball validity before the round-trip."""
    m = args.m
    rossby = -2.0 / (m + 1)
    band = omega[np.abs(omega.real) < 2.5]
    print("\n================ SANITY ================")
    print(f"  modes with |Re ω| < 2.5 (inertial band): {len(band)}")
    # Rossby anchor: real-ish, near Re=-2/(m+1), Im≈0.
    cand = band[(np.abs(band.real - rossby) < 0.12)
                & (np.abs(band.imag) < 0.12)]
    if len(cand):
        best = cand[np.argmin(np.abs(cand.real - rossby))]
        print(f"  Rossby anchor target Re=-2/{m+1}={rossby:.3f}: "
              f"nearest mode ω={best.real:+.4f}{best.imag:+.4f}j "
              f"(Re err {abs(best.real-rossby):.3f})")
    else:
        print(f"  Rossby anchor target Re={rossby:.3f}: NONE within tol "
              f"(check params/BC)")
    # Convective/growing modes (Im>0) — should exist because δ>0.
    growing = omega[omega.imag > 1e-4]
    print(f"  growing modes (Im ω > 0, expected from δ={args.delta:.0e}): "
          f"{len(growing)}"
          + (f"; max Im = {growing.imag.max():+.3f}" if len(growing) else ""))
    if len(growing):
        # Split physical inertial band from spurious high-|Re| (high-k) modes.
        # Coarse-resolution tests showed buoyancy destabilizes spurious modes
        # at |Re ω|~59 that are NOT the physical convective branch; only
        # |Re ω|<2 growing modes correspond to Bekki Fig-2c convective modes.
        gphys = growing[np.abs(growing.real) < 2.0]
        gspur = growing[np.abs(growing.real) >= 2.0]
        print(f"    -> in inertial range |Re ω|<2 (PHYSICAL): {len(gphys)}")
        print(f"    -> at |Re ω|>=2 (likely spurious high-k): {len(gspur)}"
              + (f"; sample max Im={gspur.imag.max():+.2f} at "
                 f"Re={gspur[np.argmax(gspur.imag)].real:+.1f}"
                 if len(gspur) else ""))
        for w in sorted(gphys, key=lambda z: -z.imag)[:15]:
            print(f"         ω = {w.real:+.4f} {w.imag:+.4f}j")
    print("  (Bekki Fig-2c convective anchor: Re~+0.27, Im~+0.36)")
    # Bekki: ~10% of eigenfrequencies have nonzero Im (truncation artifacts at
    # high spatial frequency). Report the fraction with |Im| not ~0.
    nontrivial_im = band[np.abs(band.imag) > 0.02]
    print(f"  inertial-band modes with |Im ω|>0.02: {len(nontrivial_im)} "
          f"({100*len(nontrivial_im)/max(len(band),1):.0f}% — Bekki sees ~10% "
          f"high-k artifacts + the physical damping branches)")
    print("========================================\n", flush=True)


def _render(omega, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print(f"[render] matplotlib unavailable ({e}); skipping PNG", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(omega.real, omega.imag, s=12, c="k", alpha=0.30,
               edgecolors="none")
    # Anchors from Fig 2b (m=8): green non-toroidal (-0.6,-0.05),
    # blue Rossby (-0.222, 0), red convective (+0.27, +0.36).
    ax.scatter([-0.6], [-0.05], s=60, c="green", zorder=5, label="non-toroidal")
    ax.scatter([-2/(args.m+1)], [0.0], s=60, c="blue", zorder=5, label="Rossby")
    ax.scatter([0.27], [0.36], s=60, c="red", zorder=5, label="convective")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.0, 1.1)
    ax.axhline(0, ls="--", c="k", lw=0.7)
    ax.axvline(0, ls="--", c="k", lw=0.7)
    ax.set_xlabel(r"$\Re[\omega]/\Omega_0$")
    ax.set_ylabel(r"$\Im[\omega]/\Omega_0$")
    ax.set_title(f"{args.backend} m={args.m} Ek={args.Ek:.2e} "
                 f"({args.N_r}×{args.N_theta})"
                 + ("" if not args.no_strain else " [no ∇lnρ·T]"))
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out + ".png", dpi=120)
    print(f"[render] {args.out}.png", flush=True)


def run_eigenmode(args):
    """Shift-invert near a target ω to pull one eigenvector, save v_r/v_θ."""
    from spdee.backends.fd2.compiled_problem import CompiledProblem

    cp, grid = build(args)
    K, M = cp.assemble_evp_matrices()
    re, im = args.eigenmode_omega
    omega_t = complex(re, im)
    sigma = 1j * omega_t           # λ = +iω (ω = −iλ)
    print(f"[eigenmode] shift-invert near ω={omega_t} (σ={sigma}) ...",
          flush=True)
    evals, evecs = CompiledProblem.solve_evp_from_matrices(
        K, M, sigma=sigma, k=args.k, return_eigenvectors=True,
    )
    om = lam_to_omega(evals)
    order = np.argsort(np.abs(om - omega_t))
    j = order[0]
    print(f"[eigenmode] nearest ω = {om[j].real:+.4f}{om[j].imag:+.4f}j",
          flush=True)
    # The state layout (which block is v_r/v_θ/v_φ/ρ1/s1) depends on the
    # backend's unknown ordering; save the raw eigenvector + grid so the
    # meridional unpacking/plotting is done downstream against the known layout.
    np.savez(
        args.out + ".npz",
        eigenvector=evecs[:, j], omega=om[j],
        omega_target=omega_t, all_omega=om,
        r=grid.r, theta=grid.theta,
        m=args.m, Ek=args.Ek, delta=args.delta,
        N_r=args.N_r, N_theta=args.N_theta, backend=args.backend,
    )
    print(f"[save] {args.out}.npz (eigenvector + grid; unpack v_r/v_θ "
          f"downstream)", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="collocated",
                   choices=["collocated", "staggered", "sh_fd", "dedalus"],
                   help="discretization backend (collocated or the "
                        "Bekki-matching staggered Arakawa-C; staggered is "
                        "complete for hydro — use --no-strain for production)")
    p.add_argument("--m", type=int, default=8, help="azimuthal order (Fig 2: 8)")
    p.add_argument("--Ek", type=float, default=EK_FIG2,
                   help="Ekman number ν/(Ω₀R²); Fig-2 caption ν=1e11 ⇒ 7.12e-6")
    p.add_argument("--delta", type=float, default=1e-6,
                   help="superadiabaticity (Fig 2: 1e-6, weakly superadiab.)")
    p.add_argument("--N_r", type=int, default=16)
    p.add_argument("--N_theta", type=int, default=72)
    p.add_argument("--order_r", type=int, default=4)
    p.add_argument("--order_theta", type=int, default=4)
    p.add_argument("--r_inner", type=float, default=0.71)
    p.add_argument("--r_outer", type=float, default=0.985)
    p.add_argument("--entropy_bc", default="neumann",
                   choices=["neumann", "dirichlet"])
    p.add_argument("--no-strain", action="store_true",
                   help="drop the ∇lnρ·T stratification term (debug only)")
    p.add_argument("--eigenmode-omega", type=float, nargs=2, default=None,
                   metavar=("RE", "IM"),
                   help="if set, pull the eigenvector near this ω instead of "
                        "computing the full spectrum")
    p.add_argument("--k", type=int, default=12,
                   help="ARPACK modes per shift in eigenmode mode")
    p.add_argument("--out", default="polytrope_spectrum",
                   help="output basename (.npz and .png)")
    args = p.parse_args(argv)

    print(f"spdee Bekki Fig-2 spectrum | backend={args.backend} m={args.m} "
          f"Ek={args.Ek:.3e} δ={args.delta:.0e} grid={args.N_r}×{args.N_theta} "
          f"strain={'off' if args.no_strain else 'on'}", flush=True)

    if args.eigenmode_omega is not None:
        run_eigenmode(args)
    else:
        run_spectrum(args)


if __name__ == "__main__":
    sys.exit(main())
