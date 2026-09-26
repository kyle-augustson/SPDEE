#!/usr/bin/env python
"""MPI shift-invert reconstruction of the Fig. 2b inertial spectrum of Bekki et al. (2022).

The dense full eigensolve (scipy.linalg.eig on the 5760² system) is slow and not
easily threaded. Instead we tile the inertial region of the complex ω-plane with
a grid of shift targets, solve a small ARPACK shift-invert problem (k eigenvalues
nearest the target) at each, parallelize the targets over MPI ranks, then
concatenate and dedup to approximate the full spectrum.

Convention: spdee uses λ = +iω, so a target frequency ω maps to shift σ = +iω,
and a returned eigenvalue λ maps back to ω = −iλ.

Workflow (mirrors inviscid_order_sweep_mpi.py):
  - rank 0 builds the target grid; every rank assembles (K, M) ONCE;
  - targets are round-robined targets[rank::size]; each rank LU-factorizes
    (K − σM) and runs ARPACK for k modes per target, streaming results to
    spectrum_rank{r}.npz;
  - after a barrier, rank 0 concatenates all ranks, dedups by eigenvalue
    proximity (eigenvalue-only), and writes the merged spectrum + a PNG.

Single-rank fallback: if mpi4py is unavailable or -n 1, runs serially.

Run (OMP/OPENBLAS/MKL_NUM_THREADS=2 per rank):
    mpiexec -n <ncores/2> python polytrope_spectrum_mpi.py --out fig2_coloc
Rank 0 writes fig2_coloc_merged.npz (+ .png).

Defaults = Fig. 2 of Bekki et al. (2022): collocated, m=8, Ek=1.44e-4, δ=1e-6, (16,72), shell
[0.71,0.985], Neumann entropy, ∇lnρ·T strain ON. Target grid spans
Re∈[-2.5,2.5]×Im∈[-2.0,1.1] (the Fig-2b axes); default spacing ~0.105 gives
~1000 targets, resolving the ~0.25-spaced damping branches.
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

# Fig. 2 caption of Bekki et al. (2022): ν=κ=1e11 cm²/s ⇒ Ek=7.12e-6 (NOT the §2.5 value
# 2e12 ⇒ 1.44e-4, which over-damps the inertial band ~20×).
EK_FIG2 = 7.123e-6


def get_comm():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD
    except Exception:
        return None


def build_targets(args):
    """Grid of ω-targets over the Fig-2b inertial window."""
    nre = int(round((args.re_max - args.re_min) / args.spacing)) + 1
    nim = int(round((args.im_max - args.im_min) / args.spacing)) + 1
    res = np.linspace(args.re_min, args.re_max, nre)
    ims = np.linspace(args.im_min, args.im_max, nim)
    RE, IM = np.meshgrid(res, ims)
    return (RE + 1j * IM).ravel()


def build_KM(args):
    from spdee.backgrounds.polytrope import PolytropeBackground
    from spdee.applications.solar_eigenmodes import make_polytrope_system
    from spdee.backends.fd2.grid import make_shell_grid
    from spdee.backends.fd2.compiled_problem import compile_problem

    poly = PolytropeBackground(r_inner=args.r_inner, r_outer=args.r_outer,
                               delta=args.delta)
    sys_ = make_polytrope_system(
        poly, m_val=args.m, Ek=args.Ek, entropy_bc=args.entropy_bc,
        geometry="shell", with_grad_lnrho_strain=not args.no_strain)
    grid = make_shell_grid(N_r=args.N_r, N_theta=args.N_theta,
                           r_inner=args.r_inner, r_outer=args.r_outer)
    cp = compile_problem(sys_, grid, m=args.m, order_r=args.order_r,
                         order_theta=args.order_theta, discretization=args.backend)
    K, M = cp.assemble_evp_matrices()
    return K, M


def solve_targets(K, M, targets, k, log_prefix="", verbose_first=False,
                  maxiter=None, sigma_eps=0.0, progress_every=100):
    """Shift-invert at each target; collect ω = −iλ. Robust to per-target fail.

    Convention: λ = +iω ⇒ ω = −iλ (Re ω = +Im λ, Im ω = −Re λ; Im ω < 0 ⇒
    damped). So a physical mode at frequency ω_t sits at λ ≈ +iω_t, and the
    shift target is σ = +iω_t (see `EvpResult.eigenvalues_to_omega`).

    ALL targets go through a maxiter-capped inline solve (not just the first),
    so a non-converging or near-singular target RAISES/skips instead of hanging
    with no output. `sigma_eps` adds a small imaginary nudge to σ to avoid
    landing exactly on an eigenvalue (which makes (K−σM) singular). The first
    target prints per-phase timing when `verbose_first`."""
    import scipy.sparse.linalg as spla
    out = []
    t0 = time.time()
    for i, omega_t in enumerate(targets):
        sigma = 1j * omega_t + 1j * sigma_eps    # λ = +iω (+ tiny nudge)
        v = verbose_first and i == 0
        try:
            ts = time.time()
            KmM = (K - sigma * M).tocsc()
            if v:
                print(f"{log_prefix}[t0] formed (K-σM) {time.time()-ts:.2f}s "
                      f"nnz {KmM.nnz}", flush=True)
            ts = time.time()
            LU = spla.splu(KmM)
            if v:
                print(f"{log_prefix}[t0] splu DONE {time.time()-ts:.2f}s "
                      f"nnz(L+U) {LU.L.nnz + LU.U.nnz}", flush=True)
            ts = time.time()
            LinOp = spla.LinearOperator(
                K.shape, matvec=lambda x: LU.solve(M @ x), dtype=complex)
            mu = spla.eigs(LinOp, k=k, which="LM",
                           return_eigenvectors=False, maxiter=maxiter)
            if v:
                print(f"{log_prefix}[t0] eigs DONE {time.time()-ts:.2f}s "
                      f"got {len(mu)}", flush=True)
            mu = mu[np.abs(mu) > 1e-12]
            lam = 1.0 / mu + sigma
            out.append(-1j * lam[np.isfinite(lam)])   # ω = −iλ
        except Exception as e:
            # ARPACK non-convergence raises ArpackNoConvergence (which carries
            # the partial eigenvalues) — salvage those, then continue.
            partial = getattr(e, "eigenvalues", None)
            if partial is not None and len(partial):
                mu = np.asarray(partial)
                mu = mu[np.abs(mu) > 1e-12]
                lam = 1.0 / mu + sigma
                out.append(-1j * lam[np.isfinite(lam)])   # ω = −iλ
                msg = f"salvaged {len(mu)}"
            else:
                msg = "no modes"
            print(f"{log_prefix}target {i} ω={omega_t:.3f} "
                  f"{type(e).__name__}: {msg}", flush=True)
        if log_prefix and (i % progress_every == 0):
            print(f"{log_prefix}{i}/{len(targets)} ({time.time()-t0:.0f}s)",
                  flush=True)
    return np.concatenate(out) if out else np.array([], dtype=complex)


def dedup(omega, tol):
    """Eigenvalue-only dedup: greedily merge points within `tol` (absolute).
    Tight tol => err toward keeping near-duplicates (over-count is harmless for
    a scatter; under-count merges real modes in the dense core)."""
    if len(omega) == 0:
        return omega
    # sort by (Re, Im) then sweep; use a grid-bucket for O(n) dedup.
    keep = []
    # bucket index by tol; check 3x3 neighborhood
    buckets = {}
    inv = 1.0 / tol
    for w in omega:
        bi = (int(np.floor(w.real * inv)), int(np.floor(w.imag * inv)))
        dup = False
        for db in ((bi[0]+dx, bi[1]+dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            for wk in buckets.get(db, ()):
                if abs(wk - w) < tol:
                    dup = True
                    break
            if dup:
                break
        if not dup:
            keep.append(w)
            buckets.setdefault(bi, []).append(w)
    return np.array(keep, dtype=complex)


def render(omega, args, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[render] skipped ({e})", flush=True)
        return
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    ax.scatter(omega.real, omega.imag, s=12, c="k", alpha=0.30, edgecolors="none")
    ax.scatter([-0.6], [-0.05], s=60, c="green", zorder=5)
    ax.scatter([-2/(args.m+1)], [0.0], s=60, c="blue", zorder=5)
    ax.scatter([0.27], [0.36], s=60, c="red", zorder=5)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.0, 1.1)
    ax.axhline(0, ls="--", c="k", lw=0.7); ax.axvline(0, ls="--", c="k", lw=0.7)
    ax.set_xlabel(r"$\Re[\omega]/\Omega_0$")
    ax.set_ylabel(r"$\Im[\omega]/\Omega_0$")
    ax.set_title(f"{args.backend} m={args.m} Ek={args.Ek:.2e} "
                 f"({args.N_r}×{args.N_theta}) shift-invert")
    fig.tight_layout(); fig.savefig(path, dpi=120)
    print(f"[render] {path}", flush=True)


def diagnostics(omega, args):
    m = args.m; rossby = -2.0 / (m + 1)
    band = omega[np.abs(omega.real) < 2.5]
    print("\n================ SANITY ================")
    print(f"  unique modes (deduped): {len(omega)}; in |Re ω|<2.5: {len(band)}")
    cand = band[(np.abs(band.real - rossby) < 0.12) & (np.abs(band.imag) < 0.12)]
    if len(cand):
        best = cand[np.argmin(np.abs(cand.real - rossby))]
        print(f"  Rossby Re=-2/{m+1}={rossby:.3f}: nearest "
              f"ω={best.real:+.4f}{best.imag:+.4f}j (err {abs(best.real-rossby):.3f})")
    else:
        print(f"  Rossby Re={rossby:.3f}: none within tol")
    growing = omega[omega.imag > 1e-4]
    print(f"  growing (Im>0, expect from δ={args.delta:.0e}): {len(growing)}"
          + (f"; max Im {growing.imag.max():+.3f}" if len(growing) else ""))
    print("========================================\n", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="collocated",
                   choices=["collocated", "staggered", "sh_fd", "dedalus"])
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--Ek", type=float, default=EK_FIG2)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument("--N_r", type=int, default=16)
    p.add_argument("--N_theta", type=int, default=72)
    p.add_argument("--order_r", type=int, default=4)
    p.add_argument("--order_theta", type=int, default=4)
    p.add_argument("--r_inner", type=float, default=0.71)
    p.add_argument("--r_outer", type=float, default=0.985)
    p.add_argument("--entropy_bc", default="neumann",
                   choices=["neumann", "dirichlet"])
    p.add_argument("--no-strain", action="store_true")
    p.add_argument("--k", type=int, default=32, help="ARPACK modes per target")
    p.add_argument("--maxiter", type=int, default=800,
                   help="cap ARPACK iterations so a non-converging/near-singular "
                        "target raises (and partial modes are salvaged) instead "
                        "of spinning forever. Default 800.")
    p.add_argument("--sigma-eps", type=float, default=1e-4,
                   help="small imaginary nudge added to σ so a target can't land "
                        "exactly on an eigenvalue (singular K−σM). Default 1e-4.")
    p.add_argument("--progress-every", type=int, default=25,
                   help="print rank-0 progress every N targets")
    p.add_argument("--spacing", type=float, default=0.105,
                   help="target grid spacing in ω (default ~1000 targets)")
    p.add_argument("--re-min", type=float, default=-2.5)
    p.add_argument("--re-max", type=float, default=2.5)
    p.add_argument("--im-min", type=float, default=-2.0)
    p.add_argument("--im-max", type=float, default=1.1)
    p.add_argument("--dedup-tol", type=float, default=1e-3,
                   help="absolute ω tol for eigenvalue-only dedup")
    p.add_argument("--merge-only", action="store_true",
                   help="skip solving; read existing <out>_rank*.npz, dedup+render "
                        "(rank 0 only; for re-merging a completed/partial run)")
    p.add_argument("--out", default="polytrope_spectrum")
    args = p.parse_args(argv)

    # --- merge-only: reconstruct from per-rank files, no MPI/solve needed ---
    if args.merge_only:
        import glob
        files = sorted(glob.glob(f"{args.out}_rank*.npz"))
        if not files:
            print(f"[merge-only] no {args.out}_rank*.npz files found", flush=True)
            return
        parts = []
        for f in files:
            w = np.load(f)["omega"]
            if len(w):
                parts.append(w)
            print(f"[merge-only] {f}: {len(w)} eigenvalues", flush=True)
        allw = np.concatenate(parts) if parts else np.array([], dtype=complex)
        print(f"[merge-only] {len(allw)} raw from {len(files)} rank files; "
              f"deduping (tol={args.dedup_tol}) ...", flush=True)
        uniq = dedup(allw, args.dedup_tol)
        np.savez(f"{args.out}_merged.npz", omega=uniq, omega_raw=allw,
                 m=args.m, Ek=args.Ek, delta=args.delta, N_r=args.N_r,
                 N_theta=args.N_theta, backend=args.backend)
        print(f"[merge-only] {len(uniq)} unique -> {args.out}_merged.npz",
              flush=True)
        diagnostics(uniq, args)
        render(uniq, args, f"{args.out}_merged.png")
        return

    comm = get_comm()
    rank = comm.Get_rank() if comm else 0
    size = comm.Get_size() if comm else 1

    if rank == 0:
        targets_all = build_targets(args)
        print(f"[setup] {len(targets_all)} targets, k={args.k}, {size} rank(s); "
              f"backend={args.backend} m={args.m} Ek={args.Ek:.3e} "
              f"({args.N_r}×{args.N_theta})", flush=True)
    else:
        targets_all = None
    if comm is not None:
        targets_all = comm.bcast(targets_all, root=0)

    t0 = time.time()
    K, M = build_KM(args)
    if rank == 0:
        print(f"[assemble] (K,M) {K.shape} nnz {K.nnz} in {time.time()-t0:.1f}s",
              flush=True)

    my_targets = targets_all[rank::size]
    my_omega = solve_targets(K, M, my_targets, args.k,
                             log_prefix=f"[rank {rank}] " if rank == 0 else "",
                             verbose_first=(rank == 0), maxiter=args.maxiter,
                             sigma_eps=args.sigma_eps,
                             progress_every=args.progress_every)
    np.savez(f"{args.out}_rank{rank}.npz", omega=my_omega)

    if comm is not None:
        comm.Barrier()
        gathered = comm.gather(my_omega, root=0)
    else:
        gathered = [my_omega]

    if rank == 0:
        allw = np.concatenate([g for g in gathered if len(g)]) if gathered else \
            np.array([], dtype=complex)
        print(f"[merge] {len(allw)} raw eigenvalues from {size} rank(s); "
              f"deduping (tol={args.dedup_tol}) ...", flush=True)
        uniq = dedup(allw, args.dedup_tol)
        np.savez(f"{args.out}_merged.npz", omega=uniq, omega_raw=allw,
                 m=args.m, Ek=args.Ek, delta=args.delta, N_r=args.N_r,
                 N_theta=args.N_theta, backend=args.backend,
                 targets=targets_all, k=args.k)
        print(f"[merge] {len(uniq)} unique modes -> {args.out}_merged.npz "
              f"(total {time.time()-t0:.0f}s)", flush=True)
        diagnostics(uniq, args)
        render(uniq, args, f"{args.out}_merged.png")


if __name__ == "__main__":
    sys.exit(main())
