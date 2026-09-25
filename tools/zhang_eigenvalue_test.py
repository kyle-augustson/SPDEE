#!/usr/bin/env python3
"""High-resolution Zhang et al. 2001 inertial-wave eigenvalue test.

Tests recovery of Zhang's analytic eigenvalues (from polynomial roots
of Eqs. 2.15 and 2.16) at user-specified resolution. Reports per-mode
relative errors and overall statistics across (m, N, parity) families.

Usage:
    python tools/zhang_eigenvalue_test.py [--N-r N_R] [--N-theta N_T] [...]

Examples:
    # Single resolution
    python tools/zhang_eigenvalue_test.py --N-r 64 --N-theta 96

    # Resolution sweep (multiple (N_r, N_theta) pairs)
    python tools/zhang_eigenvalue_test.py --sweep 32,48 48,64 64,96 96,128 128,128

    # Limit to specific m values and N range
    python tools/zhang_eigenvalue_test.py --N-r 96 --N-theta 96 \
        --m-values 1 2 --N-max 3

    # Disable twin-grid filter (use raw spectrum from one resolution)
    python tools/zhang_eigenvalue_test.py --N-r 96 --N-theta 96 --no-filter

The script computes Zhang's analytic eigenvalues by polynomial root
finding from Eqs. (2.15) symmetric / (2.16) antisymmetric. For each
target it shifts ARPACK near the expected eigenvalue, extracts the
spectrum, optionally runs the twin-grid spurious filter, and reports
how close the nearest spdee eigenvalue is.

Sign convention: spdee's `2·Cross(Ω, v) - Grad(p) = λ v` (with
Ω = ẑ) implies λ_spdee = -2iσ for Zhang's `2ẑ × u + ∇P = -2iσ u`,
so ω = 2σ = -Im(λ_spdee).

Runtime note: at N_r=24, N_theta=48 the full m∈{1,2,8} N≤2 test
takes ~5 minutes. At N_r=128, N_theta=128 expect roughly 1 hour per
resolution due to LU factorization cost of the ~75k×75k complex
matrix at each shift-invert. Reduce --m-values, --N-max, or use
--no-filter to cut the work.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Make 'spdee' importable: add the parent directory of this script's
# directory (i.e. the project root) to sys.path. Lets the script run
# from any cwd without needing pip install.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Zhang analytic eigenvalue computation
# ---------------------------------------------------------------------------

def W_Nm_symmetric_polynomial(N: int, m: int) -> np.ndarray:
    """Polynomial W^sym_{N,m}(σ) coefficients from Zhang Eq. (2.15).

    Returns coeffs array such that np.polynomial.Polynomial(coeffs)
    represents the polynomial. Real roots in [-1, 1] are the
    half-frequencies σ_{N,m,n}.
    """
    coeffs = np.zeros(2 * N + 2)
    for j in range(N + 1):
        sign = (-1) ** j
        num = math.factorial(2 * (2 * N + m - j))
        den = (math.factorial(j) * math.factorial(2 * N + m - j)
               * math.factorial(2 * (N - j)))
        prefactor = sign * num / den
        # term: prefactor · [(m+2N-2j) σ - 2(N-j)] · σ^{2(N-j)-1}
        # = prefactor (m+2N-2j) σ^{2(N-j)} - prefactor·2(N-j)·σ^{2(N-j)-1}
        p = 2 * (N - j)
        coeffs[p] += prefactor * (m + 2 * N - 2 * j)
        if p - 1 >= 0:
            coeffs[p - 1] -= prefactor * 2 * (N - j)
    nonzero = np.where(np.abs(coeffs) > 1e-300)[0]
    return coeffs[: nonzero[-1] + 1] if len(nonzero) else np.array([0.0])


def W_Nm_antisymmetric_polynomial(N: int, m: int) -> np.ndarray:
    """Polynomial W^anti_{N,m}(σ) coefficients from Zhang Eq. (2.16)."""
    coeffs = np.zeros(2 * N + 3)
    for j in range(N + 1):
        sign = (-1) ** j
        num = math.factorial(2 * (2 * N + m + 1 - j))
        den = (math.factorial(j) * math.factorial(2 * N + m + 1 - j)
               * math.factorial(2 * (N - j) + 1))
        prefactor = sign * num / den
        p = 2 * (N - j) + 1
        coeffs[p] += prefactor * (m + 2 * N - 2 * j + 1)
        coeffs[p - 1] -= prefactor * (2 * (N - j) + 1)
    nonzero = np.where(np.abs(coeffs) > 1e-300)[0]
    return coeffs[: nonzero[-1] + 1] if len(nonzero) else np.array([0.0])


def zhang_sigmas(N: int, m: int, parity: str) -> List[float]:
    """Return real roots of W_{N,m}(σ) in (-1, 1) for the given parity."""
    if parity == "SYM":
        coeffs = W_Nm_symmetric_polynomial(N, m)
    elif parity == "ANTI":
        coeffs = W_Nm_antisymmetric_polynomial(N, m)
    else:
        raise ValueError(f"parity must be 'SYM' or 'ANTI'; got {parity!r}")
    if len(coeffs) <= 1:
        return []
    roots = np.polynomial.Polynomial(coeffs).roots()
    # Keep real roots in valid range
    real = roots[np.abs(roots.imag) < 1e-10].real
    real = real[np.abs(real) < 1.0 - 1e-12]
    real = np.sort(np.unique(np.round(real, 10)))
    return list(real)


@dataclass
class ZhangTarget:
    m: int
    N: int
    parity: str  # "SYM" or "ANTI"
    sigma: float
    omega: float = field(init=False)

    def __post_init__(self):
        self.omega = 2.0 * self.sigma

    def label(self) -> str:
        return f"m={self.m} N={self.N} {self.parity} ω={self.omega:+.5f}"


def all_targets(m_values: List[int], N_max: int) -> List[ZhangTarget]:
    """Build the full list of Zhang targets for m ∈ m_values and N ≤ N_max."""
    targets = []
    for m in m_values:
        for N in range(1, N_max + 1):
            for parity in ("SYM", "ANTI"):
                for sigma in zhang_sigmas(N, m, parity):
                    targets.append(ZhangTarget(m=m, N=N, parity=parity, sigma=sigma))
    return targets


# ---------------------------------------------------------------------------
# spdee setup
# ---------------------------------------------------------------------------

def build_zhang_system():
    """Build the Zhang inertial-wave EquationSystem in spdee."""
    from spdee import (
        Unknown, Cross, Grad, Div, Equation, EquationSystem,
        BoundaryCondition, Boundary, Dot, r_hat, Background, theta,
    )
    v = Unknown.Vector("v")
    p = Unknown.Scalar("p")
    Omega = Background.Vector(
        "Omega",
        v_r=sp.cos(theta), v_theta=-sp.sin(theta), v_phi=sp.S.Zero,
    )
    eq_mom = Equation(2 * Cross(Omega, v) - Grad(p), name="momentum")
    eq_cont = Equation(Div(v), name="continuity")
    bcs = [BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vr_imp")]
    return EquationSystem([eq_mom, eq_cont], bcs, gauge_unknowns=["p"])


def search_filtered(sys_, m: int, omega_target: float,
                    N_r: int, N_theta: int, dN: int,
                    order_r: int, order_theta: int,
                    rel_tol_filter: float, k: int) -> Tuple[Optional[float], int, int]:
    """Filtered search: returns (closest_omega, n_physical, n_spurious).

    Closest_omega is None if no physical eigenvalue found near the target.
    """
    from spdee.backends.fd2.grid import make_ball_grid
    from spdee.backends.fd2.evp_filter import filter_spurious_eigenvalues

    sigma_shift = -1j * omega_target - 0.001
    try:
        phys, spur, _ = filter_spurious_eigenvalues(
            sys_,
            grid_factory=lambda nr, nt: make_ball_grid(N_r=nr, N_theta=nt, r_outer=1.0),
            N_r_base=N_r, N_theta_base=N_theta,
            m=m, k=k, sigma=sigma_shift,
            mass_equations={"momentum": "v"},
            order_r=order_r, order_theta=order_theta,
            dN=dN, rel_tolerance=rel_tol_filter,
        )
    except Exception:
        return None, 0, 0
    phys = list(phys)
    spur = list(spur)
    if not phys:
        return None, 0, len(spur)
    best = min(phys, key=lambda x: abs(-x.imag - omega_target))
    return -best.imag, len(phys), len(spur)


def search_unfiltered(sys_, m: int, omega_target: float,
                      N_r: int, N_theta: int,
                      order_r: int, order_theta: int,
                      k: int) -> Tuple[Optional[float], float, int]:
    """Unfiltered search: solve EVP once, find closest eigenvalue.

    Returns (closest_omega, |Re(λ)| of closest, n_finite_eigenvalues).
    """
    from spdee.backends.fd2.grid import make_ball_grid
    from spdee.backends.fd2.compiled_problem import compile_problem

    grid = make_ball_grid(N_r=N_r, N_theta=N_theta, r_outer=1.0)
    cp = compile_problem(sys_, grid, m=m,
                         order_r=order_r, order_theta=order_theta)
    sigma_shift = -1j * omega_target - 0.001
    res = cp.solve_evp(k=k, sigma=sigma_shift,
                       mass_equations={"momentum": "v"})
    lam = np.array(res.eigenvalues)
    finite = lam[np.isfinite(lam)]
    if len(finite) == 0:
        return None, 0.0, 0
    best = min(finite, key=lambda x: abs(-x.imag - omega_target))
    return -best.imag, abs(best.real), len(finite)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

@dataclass
class Result:
    target: ZhangTarget
    spdee_omega: Optional[float]
    rel_err: Optional[float]
    extra: dict


def run_resolution(targets: List[ZhangTarget], N_r: int, N_theta: int,
                   dN: int, order_r: int, order_theta: int,
                   rel_tol_filter: float, k: int,
                   use_filter: bool,
                   rank: int = 0, nranks: int = 1) -> List[Result]:
    """Run targets at one resolution, return results list.

    With nranks > 1, this rank handles targets where `idx % nranks == rank`
    (round-robin distribution). Targets are deterministically ordered so
    every rank sees the same partition.
    """
    sys_ = build_zhang_system()
    results = []

    # Deterministic ordering: by (m, parity, N, sigma).
    targets_sorted = sorted(targets, key=lambda t: (t.m, t.parity, t.N, t.sigma))

    # Round-robin partition.
    my_targets = [t for i, t in enumerate(targets_sorted) if i % nranks == rank]
    prefix = f"[r{rank}] " if nranks > 1 else "  "

    if nranks > 1:
        print(f"{prefix}got {len(my_targets)}/{len(targets_sorted)} targets at "
              f"N_r={N_r}, N_theta={N_theta}", flush=True)
    else:
        print(f"  {len(my_targets)} targets at N_r={N_r}, N_theta={N_theta}",
              flush=True)

    for idx, t in enumerate(my_targets, 1):
        t0 = time.time()
        print(f"{prefix}[{idx:>2}/{len(my_targets)}] "
              f"m={t.m} {t.parity} N={t.N} ω={t.omega:+.5f}... ",
              end="", flush=True)
        if use_filter:
            omega, n_phys, n_spur = search_filtered(
                sys_, m=t.m, omega_target=t.omega,
                N_r=N_r, N_theta=N_theta, dN=dN,
                order_r=order_r, order_theta=order_theta,
                rel_tol_filter=rel_tol_filter, k=k,
            )
            extra = {"n_physical": n_phys, "n_spurious": n_spur,
                     "elapsed_s": time.time() - t0}
        else:
            omega, abs_re, n_fin = search_unfiltered(
                sys_, m=t.m, omega_target=t.omega,
                N_r=N_r, N_theta=N_theta,
                order_r=order_r, order_theta=order_theta,
                k=k,
            )
            extra = {"abs_re_lambda": abs_re, "n_finite": n_fin,
                     "elapsed_s": time.time() - t0}
        if omega is None:
            rel = None
            print(f"NO ({extra['elapsed_s']:.1f}s)", flush=True)
        else:
            rel = abs(omega - t.omega) / max(abs(t.omega), 0.05)
            print(f"ω={omega:+.5f} rel={rel:.2e} ({extra['elapsed_s']:.1f}s)",
                  flush=True)
        results.append(Result(target=t, spdee_omega=omega,
                              rel_err=rel, extra=extra))
    return results


# ---------------------------------------------------------------------------
# Output / reporting
# ---------------------------------------------------------------------------

def format_results_table(results: List[Result], header: str = "") -> str:
    """Tabular formatted results, sorted by (m, parity, N, ω)."""
    out = []
    if header:
        out.append(header)
    out.append(f"{'m':>3} {'N':>2} {'par':>4} {'ω target':>11} "
               f"{'ω spdee':>11} {'rel err':>10} {'status':>6}  extra")
    out.append("-" * 80)
    results_sorted = sorted(results, key=lambda r: (r.target.m, r.target.parity,
                                                     r.target.N, r.target.omega))
    for r in results_sorted:
        t = r.target
        if r.spdee_omega is None:
            line = (f"{t.m:>3} {t.N:>2} {t.parity:>4} {t.omega:>+11.5f} "
                    f"{'—':>11} {'—':>10} {'NO':>6}")
        else:
            mark = ("★★" if r.rel_err < 1e-4 else
                    "★ " if r.rel_err < 5e-3 else
                    "? " if r.rel_err < 0.05 else
                    "✗ ")
            line = (f"{t.m:>3} {t.N:>2} {t.parity:>4} {t.omega:>+11.5f} "
                    f"{r.spdee_omega:>+11.5f} {r.rel_err:>10.2e} "
                    f"{mark:>6}")
        extra_str = ""
        if "n_physical" in r.extra:
            extra_str = (f"  phys={r.extra['n_physical']:>2} "
                         f"spur={r.extra['n_spurious']:>2}")
        elif "abs_re_lambda" in r.extra:
            extra_str = f"  |Re λ|={r.extra['abs_re_lambda']:.1e}"
        if "elapsed_s" in r.extra:
            extra_str += f"  t={r.extra['elapsed_s']:.1f}s"
        out.append(line + extra_str)
    return "\n".join(out)


def summarize(results: List[Result]) -> str:
    """Summary stats: how many in each rel-err bucket, by (parity, slow/fast)."""
    total = len(results)
    found = [r for r in results if r.spdee_omega is not None]
    not_found = total - len(found)

    buckets = {"★★ (rel<1e-4)": 0, "★ (rel<5e-3)": 0,
               "? (rel<5e-2)": 0, "✗ (rel≥5e-2)": 0, "NO (not found)": not_found}
    for r in found:
        if r.rel_err < 1e-4:
            buckets["★★ (rel<1e-4)"] += 1
        elif r.rel_err < 5e-3:
            buckets["★ (rel<5e-3)"] += 1
        elif r.rel_err < 5e-2:
            buckets["? (rel<5e-2)"] += 1
        else:
            buckets["✗ (rel≥5e-2)"] += 1

    out = [f"Total targets: {total}"]
    for k, v in buckets.items():
        pct = 100 * v / total if total else 0
        out.append(f"  {k:>20}: {v:>3} ({pct:5.1f}%)")

    # Break down by parity
    out.append("")
    out.append("By parity:")
    for parity in ("SYM", "ANTI"):
        par_results = [r for r in results if r.target.parity == parity]
        par_found = [r for r in par_results if r.spdee_omega is not None]
        par_clean = [r for r in par_found if r.rel_err < 5e-3]
        out.append(f"  {parity}: {len(par_clean)}/{len(par_results)} "
                   f"recover to rel<5e-3, {len(par_found)}/{len(par_results)} "
                   f"any match found")

    # Worst miss (largest rel err among found)
    if found:
        worst = max(found, key=lambda r: r.rel_err)
        out.append("")
        out.append(f"Worst miss: {worst.target.label()}, "
                   f"spdee ω={worst.spdee_omega:+.5f}, rel={worst.rel_err:.2e}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_sweep(s: str) -> Tuple[int, int]:
    a, b = s.split(",")
    return int(a), int(b)


def _setup_mpi(want_mpi: bool):
    """Return (comm, rank, nranks). If want_mpi is True but mpi4py is
    unavailable, exit with an error. Otherwise return a degenerate
    single-rank stub.
    """
    if not want_mpi:
        return None, 0, 1
    try:
        from mpi4py import MPI
    except ImportError:
        print("ERROR: --mpi requested but mpi4py not available", file=sys.stderr)
        sys.exit(1)
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--N-r", type=int, default=64,
                        help="Radial grid points (default 64)")
    parser.add_argument("--N-theta", type=int, default=96,
                        help="Theta grid points (default 96)")
    parser.add_argument("--sweep", type=parse_sweep, nargs="+",
                        metavar="N_R,N_T",
                        help="Resolution sweep: list of N_r,N_theta pairs "
                             "(e.g. --sweep 32,48 64,96 128,128). "
                             "Overrides --N-r/--N-theta.")
    parser.add_argument("--m-values", type=int, nargs="+", default=[1, 2, 8],
                        help="Azimuthal wavenumbers to test (default 1 2 8)")
    parser.add_argument("--N-max", type=int, default=2,
                        help="Maximum Zhang index N (default 2). Use 3 or 4 "
                             "for more modes; warning, # of targets grows fast.")
    parser.add_argument("--order-r", type=int, default=6,
                        help="Radial FD order (default 6)")
    parser.add_argument("--order-theta", type=int, default=6,
                        help="Theta FD order (default 6)")
    parser.add_argument("--dN", type=int, default=2,
                        help="Twin-grid resolution offset (default 2). "
                             "Smaller dN = stricter filter.")
    parser.add_argument("--rel-tol-filter", type=float, default=2e-3,
                        help="Twin-grid filter relative tolerance (default 2e-3)")
    parser.add_argument("--k", type=int, default=8,
                        help="ARPACK k (number of eigenvalues per shift) (default 8)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip the twin-grid filter; use raw spectrum")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-resolution table; only show summary")
    parser.add_argument("--mpi", action="store_true",
                        help="Parallelize targets over MPI ranks (requires "
                             "mpi4py; launch with mpirun/mpiexec). Each rank "
                             "handles a round-robin subset of targets. "
                             "Throttle OpenMP per-rank via OMP_NUM_THREADS in "
                             "the launcher to avoid oversubscription.")
    args = parser.parse_args()

    comm, rank, nranks = _setup_mpi(args.mpi)
    is_root = (rank == 0)

    # Build target list (every rank does this independently; deterministic).
    if is_root:
        print(f"Building Zhang targets for m ∈ {args.m_values}, N ≤ {args.N_max}...")
    targets = all_targets(args.m_values, args.N_max)
    if is_root:
        print(f"  Total: {len(targets)} (m, N, parity, σ) targets")
        if nranks > 1:
            print(f"  MPI: {nranks} ranks (round-robin over targets)")
        print()

    # Decide resolutions
    if args.sweep:
        resolutions = args.sweep
    else:
        resolutions = [(args.N_r, args.N_theta)]

    overall_t0 = time.time()
    for N_r, N_theta in resolutions:
        if is_root:
            print("=" * 80)
            print(f"Resolution: N_r={N_r}, N_theta={N_theta}, "
                  f"order_r={args.order_r}, order_theta={args.order_theta}, "
                  f"filter={'OFF' if args.no_filter else 'ON'}")
            print("=" * 80)
        t0 = time.time()

        # Each rank runs its subset of targets.
        my_results = run_resolution(
            targets, N_r=N_r, N_theta=N_theta,
            dN=args.dN, order_r=args.order_r, order_theta=args.order_theta,
            rel_tol_filter=args.rel_tol_filter, k=args.k,
            use_filter=(not args.no_filter),
            rank=rank, nranks=nranks,
        )

        # Gather results to root for reporting.
        if comm is not None:
            comm.Barrier()
            all_results_lists = comm.gather(my_results, root=0)
            if is_root:
                results = []
                for lst in all_results_lists:
                    results.extend(lst)
            else:
                results = None
        else:
            results = my_results

        elapsed = time.time() - t0
        if is_root:
            if not args.quiet:
                print()
                print(format_results_table(results))
            print()
            print(summarize(results))
            print(f"\nResolution elapsed: {elapsed:.1f} s")
            print()

    if is_root:
        print("=" * 80)
        print(f"Total elapsed: {time.time() - overall_t0:.1f} s")
        print("=" * 80)


if __name__ == "__main__":
    main()
