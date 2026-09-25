#!/usr/bin/env python3
"""Parallel δ-sweep of the Bekki Figure-2 convective mode.

Each MPI rank runs an independent dense eigensolve at one (or more)
superadiabaticity value δ and reports the most-growing prograde mode in the
Re ∈ (0.05, 0.6) box, the columnar convective mode (Bekki Fig. 2c anchor
Re ≈ +0.27, Im ≈ +0.36), so that Im(ω) can be read as a function of δ.

Run (one rank per δ; each rank pins to a single core, its own serial dense
solve, so the solves run concurrently):

    mpiexec -n 5 \
        python3 scripts/delta_sweep_mpi.py

Options:
    --deltas 1e-6,1e-5,1e-4,1e-3,1e-2   comma list of δ values (default these 5)
    --Nr 16 --Ntheta 72                 grid (default Bekki Fig-2)
    --Ek 7.123e-6                       Ekman number (default Fig-2 caption)
    --m 8                               azimuthal order
    --re-lo 0.05 --re-hi 0.6            candidate Re-box
    --out delta_sweep.npz               save full per-δ growing-mode lists

Each rank holds a ~2.3 GB dense working set at (16, 72). argmax(Im) in the
Re box is the convective mode at small δ (1e-6..1e-4); at large δ the box
may catch a different unstable mode.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import scipy.linalg as sla

try:
    from mpi4py import MPI
    _HAVE_MPI = True
except Exception:  # allow serial fallback
    _HAVE_MPI = False

from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spdee.backgrounds.polytrope import PolytropeBackground
from spdee.applications.solar_eigenmodes import make_polytrope_system
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem


def solve_one(delta, args, grid):
    """Dense eigensolve at one δ; return (delta, candidate ω or None,
    array of all growing-in-box ω)."""
    poly = PolytropeBackground(delta=delta)
    sys_ = make_polytrope_system(poly, m_val=args.m, Ek=args.Ek)
    cp = compile_problem(sys_, grid, m=args.m,
                         order_r=args.order_r, order_theta=args.order_theta)
    K, M = cp.assemble_evp_matrices()
    lam = sla.eig(K.toarray(), M.toarray(), right=False)
    lam = lam[np.isfinite(lam)]
    lam = lam[np.abs(lam) < 1e8]
    om = -1j * lam
    box = om[(om.real > args.re_lo) & (om.real < args.re_hi)
             & (om.imag > -0.5) & (om.imag < 1.0)]
    cand = box[np.argmax(box.imag)] if len(box) else None
    return delta, cand, box


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deltas", default="1e-6,1e-5,1e-4,1e-3,1e-2")
    p.add_argument("--Nr", type=int, default=16)
    p.add_argument("--Ntheta", type=int, default=72)
    p.add_argument("--Ek", type=float, default=7.123e-6)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--order_r", type=int, default=4)
    p.add_argument("--order_theta", type=int, default=4)
    p.add_argument("--r_inner", type=float, default=0.71)
    p.add_argument("--r_outer", type=float, default=0.985)
    p.add_argument("--re-lo", type=float, default=0.05)
    p.add_argument("--re-hi", type=float, default=0.6)
    p.add_argument("--out", default="delta_sweep.npz")
    args = p.parse_args(argv)

    deltas = [float(x) for x in args.deltas.split(",")]

    if _HAVE_MPI:
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, size = None, 0, 1

    grid = make_shell_grid(N_r=args.Nr, N_theta=args.Ntheta,
                           r_inner=args.r_inner, r_outer=args.r_outer)

    if rank == 0:
        print(f"[setup] δ-sweep {deltas}  m={args.m} Ek={args.Ek:.3e} "
              f"({args.Nr}x{args.Ntheta})  {size} rank(s)", flush=True)

    # Round-robin δ over ranks.
    mine = []
    for i in range(rank, len(deltas), size):
        d = deltas[i]
        t0 = time.time()
        res = solve_one(d, args, grid)
        mine.append(res)
        cand = res[1]
        msg = (f"Re={cand.real:+.4f} Im={cand.imag:+.4f}"
               if cand is not None else "(no candidate)")
        print(f"[rank {rank}] δ={d:.0e}: {msg}  "
              f"({time.time()-t0:.0f}s, {len(res[2])} growing in box)",
              flush=True)

    allres = comm.gather(mine, root=0) if _HAVE_MPI else [mine]

    if rank == 0:
        flat = sorted([r for chunk in allres for r in chunk],
                      key=lambda t: t[0])
        print("\n================ δ-SWEEP ================", flush=True)
        print(" delta      Re(ω)     Im(ω)    #growing-in-box")
        save = {}
        for d, cand, box in flat:
            if cand is not None:
                print(f"  {d:.0e}   {cand.real:+.4f}   {cand.imag:+.4f}   "
                      f"{len(box)}")
            else:
                print(f"  {d:.0e}   (none)")
            save[f"box_{d:.0e}"] = box
        # slope at the small-δ end (1e-6 -> 1e-4) if available
        sm = [(d, c.imag) for d, c, _ in flat if c is not None and d <= 1e-4]
        if len(sm) >= 2:
            (d0, i0), (d1, i1) = sm[0], sm[-1]
            slope = (i1 - i0) / (np.log10(d1) - np.log10(d0))
            print(f"\n  dIm/dlog10(δ) over [{d0:.0e},{d1:.0e}] = {slope:+.3f}")
            print("  (large slope ⇒ near threshold, form OK; "
                  "~0 ⇒ structurally decoupled)")
        print("  Bekki Fig-2c anchor: Re~+0.27, Im~+0.36", flush=True)
        np.savez(args.out, deltas=np.array(deltas), **save)
        print(f"\n[save] {args.out}", flush=True)


if __name__ == "__main__":
    main()
