#!/usr/bin/env python3
"""Twin-grid convergence test of the Bekki Figure-2 modes on the
collocated core.

`filter_spurious_eigenvalues` solves the EVP on (N_r, N_θ) and
(N_r + dN, N_θ + dN) and keeps the modes whose λ converges between the two
grids: a physical mode converges, a checkerboard mode of the collocated
grid drifts O(1) and is rejected. Three shift targets, one per MPI rank:

  rank 0: convective band   ω ≈ +0.30 + 0.0j
  rank 1: Rossby anchor     ω ≈ -0.222 + 0.0j   (a physical control)
  rank 2: non-toroidal      ω ≈ -0.60 - 0.05j   (a physical control)

Run:
    mpiexec -n 3 \
        python3 scripts/checkerboard_twin_grid_mpi.py

Options:
    --Nr 16 --Ntheta 72   base grid (default Fig-2)
    --dN 2                refinement increment (default 2; larger = clearer
                          drift signal for marginal modes)
    --k 24                eigenvalues per shift-invert solve
    --Ek 7.123e-6 --delta 1e-6 --m 8
    --reltol 1e-2         convergence tolerance (mode kept if λ matches to this)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path as _Path

import numpy as np

try:
    from mpi4py import MPI
    _HAVE_MPI = True
except Exception:
    _HAVE_MPI = False

_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from spdee.backgrounds.polytrope import PolytropeBackground
from spdee.applications.solar_eigenmodes import make_polytrope_system
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.evp_filter import filter_spurious_eigenvalues


# (label, target ω). σ = +iω in the spdee convention (λ = +iω).
TARGETS = [
    ("convective", 0.30 + 0.00j),
    ("rossby",    -0.222 + 0.00j),
    ("nontoroidal", -0.60 - 0.05j),
]


def run_target(label, omega_t, args):
    poly = PolytropeBackground(delta=args.delta)
    system = make_polytrope_system(poly, m_val=args.m, Ek=args.Ek)

    def grid_factory(nr, nt):
        return make_shell_grid(N_r=nr, N_theta=nt,
                               r_inner=args.r_inner, r_outer=args.r_outer)

    sigma = 1j * omega_t          # λ = +iω
    t0 = time.time()
    phys, spur_base, spur_fine = filter_spurious_eigenvalues(
        system, grid_factory,
        N_r_base=args.Nr, N_theta_base=args.Ntheta,
        m=args.m, k=args.k, sigma=sigma,
        order_r=args.order_r, order_theta=args.order_theta,
        dN=args.dN, rel_tolerance=args.reltol,
    )
    dt = time.time() - t0
    phys_om = -1j * np.asarray(phys)            # ω = −iλ
    spur_om = -1j * np.asarray(spur_base)

    # Did a mode near the target survive as PHYSICAL?
    def nearest(arr):
        if not len(arr):
            return None
        return arr[np.argmin(np.abs(arr - omega_t))]
    near_phys = nearest(phys_om)
    near_spur = nearest(spur_om)
    return {
        "label": label, "omega_t": omega_t, "dt": dt,
        "n_phys": len(phys_om), "n_spur": len(spur_om),
        "near_phys": near_phys, "near_spur": near_spur,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--Nr", type=int, default=16)
    p.add_argument("--Ntheta", type=int, default=72)
    p.add_argument("--dN", type=int, default=2)
    p.add_argument("--k", type=int, default=24)
    p.add_argument("--Ek", type=float, default=7.123e-6)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--order_r", type=int, default=4)
    p.add_argument("--order_theta", type=int, default=4)
    p.add_argument("--r_inner", type=float, default=0.71)
    p.add_argument("--r_outer", type=float, default=0.985)
    p.add_argument("--reltol", type=float, default=1e-2)
    args = p.parse_args(argv)

    if _HAVE_MPI:
        comm = MPI.COMM_WORLD
        rank, size = comm.Get_rank(), comm.Get_size()
    else:
        comm, rank, size = None, 0, 1

    if rank == 0:
        print(f"[setup] twin-grid convergence test, base {args.Nr}x{args.Ntheta}"
              f" vs {args.Nr+args.dN}x{args.Ntheta+args.dN}, k={args.k}, "
              f"reltol={args.reltol}, Ek={args.Ek:.3e} delta={args.delta:.0e}",
              flush=True)

    mine = []
    for i in range(rank, len(TARGETS), size):
        label, omega_t = TARGETS[i]
        r = run_target(label, omega_t, args)
        mine.append(r)
        nph = (f"{r['near_phys'].real:+.4f}{r['near_phys'].imag:+.4f}j"
               if r['near_phys'] is not None else "—")
        print(f"[rank {rank}] {label} (target {omega_t:+.3f}): "
              f"nearest PHYSICAL ω={nph}  "
              f"({r['n_phys']} phys / {r['n_spur']} spurious, {r['dt']:.0f}s)",
              flush=True)

    allres = comm.gather(mine, root=0) if _HAVE_MPI else [mine]

    if rank == 0:
        flat = [r for chunk in allres for r in chunk]
        print("\n============ CHECKERBOARD / CONVERGENCE TEST ============",
              flush=True)
        for r in sorted(flat, key=lambda x: x["label"]):
            t = r["omega_t"]
            npr = r["near_phys"]
            nsp = r["near_spur"]
            print(f"\n  {r['label']}  (target ω = {t:+.3f})")
            if npr is not None and abs(npr - t) < 0.25:
                print(f"    PHYSICAL mode near target: ω = "
                      f"{npr.real:+.4f}{npr.imag:+.4f}j  -> SURVIVES "
                      f"convergence (grid-converged)")
            else:
                near = f"{npr.real:+.4f}{npr.imag:+.4f}j" if npr is not None else "none"
                print(f"    NO converged physical mode within 0.25 of target "
                      f"(nearest physical: {near})")
                if nsp is not None and abs(nsp - t) < 0.25:
                    print(f"    but a SPURIOUS (grid-drifting) mode sits near "
                          f"target: ω = {nsp.real:+.4f}{nsp.imag:+.4f}j")
                    print("    => consistent with CHECKERBOARD: the mode here "
                          "is grid-dependent, not physical.")
        print("\n  Read: 'rossby' and 'nontoroidal' are physical controls and "
              "MUST survive.")
        print("  If they survive but 'convective' does NOT, the convective "
              "candidate is")
        print("  spurious/checkerboard-corrupted on the collocated grid -> "
              "mimetic rebuild justified.")
        print("=========================================================",
              flush=True)


if __name__ == "__main__":
    main()
