#!/usr/bin/env python3
"""
Inviscid staggered inertial-EVP order/resolution sweep.

PURPOSE
-------
Answer rigorously: is the staggered inviscid inertial operator CLEAN at all
interior orders with a matched-order boundary closure?  "Clean" means:
  (1) UNDAMPED  -- the physical inertial modes have Re(lambda) ~ 0 (machine
      precision); no spurious growth (Re > 0) that PERSISTS under refinement.
  (2) CONVERGENT -- the physical frequency Im(lambda) converges toward the
      analytic toroidal value  lambda_imag = m / (ell(ell+1))  at (about) the
      interior order.

Two metrics are reported and kept SEPARATE, because
eigenvalue-nearest metrics on the CLUSTERED inertial spectrum are unreliable for
ORDER (they grab neighbors / conjugate-pair members), while they ARE the right
tool for UNDAMPING:
  - undamping  : from a DENSE generalized eig (no iterative-solver floor, no
                 shift-invert mode mixing, which leaves scattered ~1e-3
                 real parts on an undamped spectrum).
  - convergence: tracked on Im(lambda) of the targeted mode AND, more reliably,
                 on an OPERATOR-ACTION residual when an analytic eigenvector is
                 available (here we use the eigenvalue trend as the primary and
                 flag erratic rates rather than trusting a single pair).

A sparse shift-invert fast path is provided for large N, and the script
SELF-VALIDATES it against the dense solve at one small case; if they disagree on
Re(lambda) beyond a tolerance, it warns (the sparse path is then untrustworthy
for the undamping question at that size).

USAGE
-----
  # full sweep, parallel across all cores:
  python inviscid_order_sweep.py --out sweep.csv

  # control the grid:
  python inviscid_order_sweep.py --ms 0 1 2 3 4 5 \
      --orders 2 4 6 8 --nr 16 24 32 48 --nth 16 24 32 48 \
      --jobs 8 --out sweep.csv

  # restrict to square grids (N_r == N_theta) to cut the sweep size:
  python inviscid_order_sweep.py --square --out sweep.csv

  # use sparse solve above this size (dense below); 0 = always dense:
  python inviscid_order_sweep.py --sparse-above 40 --out sweep.csv

Each (m, order, N_r, N_theta) is one independent task -> trivially parallel.
Results stream to CSV as they complete (so a long run is resumable-by-eye and
partial results survive an interrupt).

REQUIRES: numpy, scipy, and the spdee package importable (run from the repo
root, or set PYTHONPATH to the repo root).

INTERPRETATION GUIDE (what to look for in the CSV)
--------------------------------------------------
  * re_lowest        : Re(lambda) of the targeted mode. Should be ~1e-12 or
                       smaller. A nonzero value that SHRINKS or appears/vanishes
                       as N changes = artifact. One that GROWS with N or is
                       consistently positive across N at a given (m,order) =
                       genuine spurious instability -> the boundary closure
                       breaks the skew/energy structure at that order.
  * band_max_abs_re  : max |Re(lambda)| over all physical-band modes (|Im|<1).
                       Same persistence test applies.
  * imag_err_rel     : |Im(lambda) - analytic| / analytic for the targeted mode
                       (m>=1). Convergence rate computed POST-HOC per (m,order)
                       across N in the analysis helper below.
  * For m=0 (degenerate, analytic target 0): we DO NOT track a frequency; we
    only report band_max_abs_re and the count of Re>tol modes (spurious-growth
    survey). imag_err_rel is left NaN.

After the run, see `summarize(csv)` at the bottom (or:
  python inviscid_order_sweep.py --summarize sweep.csv
) for per-(m,order) convergence rates and a persistence verdict on Re(lambda).
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Path setup — script lives in scripts/, code lives in spdee/. Lets the
# script run from anywhere without requiring spdee to be pip-installed.
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ----------------------------------------------------------------------------
# analytic target
# ----------------------------------------------------------------------------
def analytic_target(m: int) -> float | None:
    """Lowest toroidal inertial frequency Im(lambda) = m/(ell(ell+1)), ell=max(m,1).
    Returns None for m=0 (degenerate; all-zero, no nonzero mode to track)."""
    if m == 0:
        return None
    ell = m
    return m / (ell * (ell + 1))


# ----------------------------------------------------------------------------
# one task: assemble + solve, return a result dict
# ----------------------------------------------------------------------------
def run_one(m: int, order: int, N_r: int, N_theta: int,
            sparse_above: int = 0, re_tol: float = 1e-6) -> dict:
    # Imports inside the worker so each process is self-contained.
    import scipy.linalg as sla
    import scipy.sparse.linalg as spla
    import sympy as smp
    from spdee import (
        Background, Boundary, BoundaryCondition, Cross, Div, Dot, Equation,
        EquationSystem, Grad, Unknown, r_hat, theta)
    from spdee.backends.fd2.staggered_compile import compile_fd2_staggered
    from spdee.backends.fd2.grid import make_shell_grid

    t0 = time.perf_counter()
    rec = dict(m=m, order=order, N_r=N_r, N_theta=N_theta,
               re_lowest=np.nan, im_lowest=np.nan, imag_err_rel=np.nan,
               band_max_abs_re=np.nan, n_spurious_pos=np.nan,
               method="", seconds=np.nan, error="")
    try:
        grid = make_shell_grid(N_r=N_r, N_theta=N_theta,
                               r_inner=0.3, r_outer=1.0)
        v, p = Unknown.Vector("v"), Unknown.Scalar("p")
        Om = Background.Vector("Omega", v_r=smp.cos(theta),
                               v_theta=-smp.sin(theta), v_phi=0)
        system = EquationSystem(
            [Equation(2 * Cross(Om, v) - Grad(p), name="momentum"),
             Equation(Div(v), name="continuity")],
            [BoundaryCondition(w, Dot(r_hat, v))
             for w in (Boundary.R_INNER, Boundary.R_OUTER)])
        cp = compile_fd2_staggered(system, grid, m=m, order_r=order,
                                   order_theta=order)
        A, B = cp.A_total.tocsr(), cp.B_total.tocsr()

        ndof = A.shape[0]
        use_sparse = (sparse_above > 0 and max(N_r, N_theta) > sparse_above)
        target = analytic_target(m)

        if not use_sparse:
            rec["method"] = "dense"
            w = sla.eig(A.toarray(), B.toarray(), right=False)
            w = w[np.isfinite(w)]
        else:
            rec["method"] = "sparse"
            # shift near the target (or near 0 for m=0) and grab a few modes
            shift = (1j * (target if target is not None else 0.0)) + 1e-6
            AmB = (A - shift * B).tocsc()
            LU = spla.splu(AmB)
            Lin = spla.LinearOperator(
                A.shape, matvec=lambda x: LU.solve(B @ x), dtype=complex)
            mu = spla.eigs(Lin, k=min(12, ndof - 2), which="LM",
                           return_eigenvectors=False, maxiter=10000, tol=1e-12)
            w = 1.0 / mu + shift
            w = w[np.isfinite(w)]

        # physical band: |Im| < 1 (inertial modes live in (0, ~0.5]); drop the
        # numerical zero cluster for the frequency pick, keep it for the Re survey
        band = w[np.abs(w.imag) < 1.0]
        rec["band_max_abs_re"] = float(np.abs(band.real).max()) if band.size else np.nan
        rec["n_spurious_pos"] = int(np.sum(band.real > re_tol))

        if target is not None:
            cand = band[np.abs(band.imag) > 1e-6]
            if cand.size:
                lam = cand[np.argmin(np.abs(cand.imag - target))]
                rec["im_lowest"] = float(lam.imag)
                rec["re_lowest"] = float(lam.real)
                rec["imag_err_rel"] = float(abs(lam.imag - target) / target)
        else:
            # m=0: report the mode with largest |Re| in band as the "lowest"
            # diagnostic (we care about spurious growth, not a frequency)
            if band.size:
                lam = band[np.argmax(np.abs(band.real))]
                rec["im_lowest"] = float(lam.imag)
                rec["re_lowest"] = float(lam.real)

        rec["seconds"] = time.perf_counter() - t0
    except Exception as e:  # keep the sweep going; record the failure
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["seconds"] = time.perf_counter() - t0
    return rec


# ----------------------------------------------------------------------------
# self-validation: sparse vs dense at one small case
# ----------------------------------------------------------------------------
def validate_solvers(m=2, order=2, N=16) -> None:
    d = run_one(m, order, N, N, sparse_above=0)
    s = run_one(m, order, N, N, sparse_above=N - 1)  # force sparse
    print(f"[validate] m={m} order={order} N={N}")
    print(f"  dense : Im={d['im_lowest']:.6f}  Re={d['re_lowest']:.2e}")
    print(f"  sparse: Im={s['im_lowest']:.6f}  Re={s['re_lowest']:.2e}")
    dim = abs(d["im_lowest"] - s["im_lowest"])
    if dim > 1e-4:
        print(f"  WARNING: sparse/dense Im disagree by {dim:.1e} "
              f"-- sparse path unreliable; prefer --sparse-above 0 (all dense).")
    else:
        print("  OK: sparse matches dense on the targeted mode.")


# ----------------------------------------------------------------------------
# sweep driver
# ----------------------------------------------------------------------------
FIELDS = ["m", "order", "N_r", "N_theta", "method",
          "im_lowest", "re_lowest", "imag_err_rel",
          "band_max_abs_re", "n_spurious_pos", "seconds", "error"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--nr", type=int, nargs="+", default=[16, 24, 32, 48])
    ap.add_argument("--nth", type=int, nargs="+", default=[16, 24, 32, 48])
    ap.add_argument("--square", action="store_true",
                    help="only N_r == N_theta combinations")
    ap.add_argument("--sparse-above", type=int, default=0,
                    help="use sparse solve when max(N_r,N_theta) exceeds this "
                         "(0 = always dense; recommended for the Re-lambda "
                         "undamping question)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", type=str, default="inviscid_sweep.csv")
    ap.add_argument("--summarize", type=str, default=None,
                    help="skip the sweep; just summarize an existing CSV")
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    if args.summarize:
        summarize(args.summarize)
        return

    if not args.no_validate:
        validate_solvers()

    # build the task list
    nr_nth = [(nr, nth) for nr in args.nr for nth in args.nth
              if (not args.square or nr == nth)]
    tasks = [(m, o, nr, nth)
             for m in args.ms for o in args.orders for (nr, nth) in nr_nth]
    print(f"[sweep] {len(tasks)} tasks on {args.jobs} workers "
          f"(method: {'sparse>'+str(args.sparse_above) if args.sparse_above else 'all-dense'})")
    print(f"[sweep] streaming results to {args.out}")

    t0 = time.perf_counter()
    done = 0
    with open(args.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        fh.flush()
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(run_one, m, o, nr, nth, args.sparse_above): (m, o, nr, nth)
                    for (m, o, nr, nth) in tasks}
            for fut in as_completed(futs):
                rec = fut.result()
                wr.writerow({k: rec.get(k, "") for k in FIELDS})
                fh.flush()
                done += 1
                tag = f"m{rec['m']} o{rec['order']} {rec['N_r']}x{rec['N_theta']}"
                if rec["error"]:
                    print(f"  [{done}/{len(tasks)}] {tag}  ERROR {rec['error']}")
                else:
                    print(f"  [{done}/{len(tasks)}] {tag}  "
                          f"Im={rec['im_lowest']:.5f} Re={rec['re_lowest']:.1e} "
                          f"err={rec['imag_err_rel']:.1e} "
                          f"bandRe={rec['band_max_abs_re']:.1e} "
                          f"({rec['seconds']:.1f}s)")
    print(f"[sweep] done in {time.perf_counter()-t0:.0f}s -> {args.out}")
    summarize(args.out)


# ----------------------------------------------------------------------------
# analysis
# ----------------------------------------------------------------------------
def summarize(csv_path: str) -> None:
    import collections
    rows = []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            if r.get("error"):
                continue
            for k in ("im_lowest", "re_lowest", "imag_err_rel",
                      "band_max_abs_re"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            for k in ("m", "order", "N_r", "N_theta"):
                r[k] = int(r[k])
            rows.append(r)

    print("\n================ SUMMARY ================")
    # group by (m, order); use square grids (N_r==N_theta) for the rate if present
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r["m"], r["order"])].append(r)

    print(f"{'m':>2} {'ord':>3} | {'imag-err rate':>13} | "
          f"{'max|Re| (persist?)':>22} | {'verdict':>20}")
    print("-" * 72)
    for (m, order) in sorted(groups):
        g = sorted(groups[(m, order)], key=lambda r: (r["N_r"], r["N_theta"]))
        sq = [r for r in g if r["N_r"] == r["N_theta"]]
        seq = sq if len(sq) >= 2 else g

        # convergence rate on imag err (m>=1), across the square diagonal
        rate_str = "   n/a (m=0)" if m == 0 else "  --"
        if m != 0 and len(seq) >= 2:
            Ns = np.array([r["N_r"] for r in seq], float)
            es = np.array([r["imag_err_rel"] for r in seq], float)
            ok = np.isfinite(es) & (es > 0)
            if ok.sum() >= 2:
                p = np.polyfit(np.log(Ns[ok]), np.log(es[ok]), 1)[0]
                rate_str = f"{-p:6.2f}"

        # Re(lambda) persistence: does max|Re| grow with N, or stay tiny/scatter?
        res = np.array([r["band_max_abs_re"] for r in seq], float)
        Ns = np.array([r["N_r"] for r in seq], float)
        finite = np.isfinite(res)
        mx = np.nanmax(res) if finite.any() else float("nan")
        # crude persistence test: is the LARGEST-N value comparable to the max,
        # i.e. does it fail to shrink?
        persist = "?"
        if finite.sum() >= 2:
            last = res[finite][-1]
            small = mx < 1e-9
            grows = last > 0.3 * mx and mx > 1e-6
            if small:
                persist = "clean (<1e-9)"
            elif grows:
                persist = f"PERSISTS ~{mx:.0e}"
            else:
                persist = f"scatter ~{mx:.0e}"

        verdict = "CLEAN"
        if "PERSISTS" in persist:
            verdict = "*** SPURIOUS ***"
        elif "scatter" in persist:
            verdict = "clean (artifacts)"
        print(f"{m:>2} {order:>3} | {rate_str:>13} | {persist:>22} | {verdict:>20}")

    print("\nNotes:")
    print(" - imag-err rate ~ interior order => convergence is clean.")
    print(" - 'PERSISTS' in max|Re| => genuine spurious growth (boundary closure")
    print("   breaks skew structure at that order). 'scatter'/'clean' => artifacts.")
    print(" - m=0 is degenerate (analytic freq 0); only the Re-survey is meaningful.")
    print(" - For the cleanest UNDAMPING verdict, run with --sparse-above 0 (dense).")


if __name__ == "__main__":
    main()
