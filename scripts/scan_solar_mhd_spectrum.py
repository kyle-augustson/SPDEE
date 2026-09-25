#!/usr/bin/env python3
"""Bekki + dipolar B₀ : spectrum scan for a Figure-2-style plot.

For each m, scan a fine grid of oscillatory-frequency targets across
the inertial band ω/Ω₀ ∈ (ω_min, ω_max), solving for `k` eigenvalues
near each target. The union of all per-target solves traces out the
full spectrum in the band. Repeat across Lehnert numbers to show the
magnetic dependence.

This mirrors the parallelization in the reference Dedalus script:

  * **Outer loop over m**: one rank assembles the (K, M) matrices for
    each (m, Le) and saves them to disk; all ranks then read them.
  * **Inner loop over targets**: the target grid is divided among the
    ranks (round-robin), each doing LU + ARPACK per target.

The matrices are σ-INDEPENDENT, so assembling once and reusing across
the 5000 targets saves both the assembly time and (with the on-disk
share) the memory of every rank rebuilding them.

Convention
----------
spdee uses λ = −iω (time dependence e^{λt} = e^{−iωt}), so a target at
oscillatory frequency ω_t with slight damping γ_t maps to the shift

    σ = γ_t − i·ω_t

The scan parameter is the physical ω_t (real); we set a fixed small
γ_t = −0.001 so the shift sits just inside the damped half-plane (the
same trick as the Dedalus script's −0.001 real offset). Eigenvalues
are reported as (ω = −Im(λ), γ = −Re(λ)).

Usage
-----

  # MPI, the production way (matches the Dedalus script's pattern):
  mpiexec -n 32 python scripts/scan_solar_mhd_spectrum.py \
      --output-dir results/spectrum \
      --m-values 1 2 3 4 \
      --Le-values 0 0.05 \
      --omega-min -2 --omega-max 2 --n-targets 5001 \
      --k 64 --N-r 128 --N-theta 128

  # Serial first-light (no MPI): 1 m, 1 Le, coarse target grid.
  python scripts/scan_solar_mhd_spectrum.py --first-light

If mpi4py is unavailable or the script is launched without mpiexec,
it runs serially (rank 0 of 1) — correct, just not parallel.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
for p in (_REPO_ROOT, _REPO_ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


# ---------------------------------------------------------------------------
# MPI bootstrap (graceful fallback to serial)
# ---------------------------------------------------------------------------

def _get_mpi():
    """Return (comm, rank, size). Falls back to a serial stub."""
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        return comm, comm.rank, comm.size
    except Exception:
        class _SerialComm:
            rank = 0
            size = 1
            def Barrier(self): pass
            def bcast(self, obj, root=0): return obj
        c = _SerialComm()
        return c, 0, 1


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_M_VALUES = [1, 2, 3, 4]
DEFAULT_LE_VALUES = [0.0, 0.05]
DEFAULT_OMEGA_MIN = -2.0
DEFAULT_OMEGA_MAX = 2.0
DEFAULT_N_TARGETS = 5001
DEFAULT_GAMMA_TARGET = -0.001    # small damping offset on the shift
DEFAULT_K = 64
DEFAULT_EK = 1e-4
DEFAULT_PM = 1.0
DEFAULT_AMP_RATIO = 30000.0
DEFAULT_R_INNER = 0.71
DEFAULT_R_OUTER = 0.985
DEFAULT_N_R = 128
DEFAULT_N_T = 128
DEFAULT_ORDER_R = 4
DEFAULT_ORDER_T = 4


# ---------------------------------------------------------------------------
# Matrix assembly + disk sharing
# ---------------------------------------------------------------------------

def _matrix_paths(out_dir: Path, m: int, Le: float) -> Tuple[Path, Path]:
    tag = "Le0" if Le == 0 else f"Le{Le:.4e}".replace("-", "n").replace(".", "_").replace("+", "p")
    base = out_dir / "_matrices"
    return base / f"m{m:02d}_{tag}_K.npz", base / f"m{m:02d}_{tag}_M.npz"


def _assemble_and_save(
    m: int, Le: float, out_dir: Path,
    Ek: float, Pm: float, amp_ratio: float,
    r_inner: float, r_outer: float,
    N_r: int, N_theta: int, order_r: int, order_theta: int,
    data_dir: str,
) -> Tuple[Path, Path, int]:
    """Compile the (m, Le) problem, assemble (K, M), save to disk.

    Returns (K_path, M_path, state_size). Called by the assembler rank
    only.
    """
    from spdee.backgrounds.solar import SolarBackgrounds
    from spdee.backends.fd2.grid import make_shell_grid
    from spdee.backends.fd2.compiled_problem import compile_problem
    from spdee.applications.solar_eigenmodes import make_solar_shell_system

    solar = SolarBackgrounds(
        Path(data_dir), ('shell', r_inner, r_outer),
        nondimensional=True, uniform_core_blend=True,
    )
    sys_ = make_solar_shell_system(
        solar, m_val=m, Ek=Ek, Le=Le, Pm=Pm,
        amplitude_ratio=amp_ratio, entropy_bc="neumann",
        geometry="shell", r_inner=r_inner, r_outer=r_outer,
    )
    grid = make_shell_grid(
        N_r=N_r, N_theta=N_theta, r_inner=r_inner, r_outer=r_outer,
    )
    cp = compile_problem(sys_, grid, m=m, order_r=order_r, order_theta=order_theta)
    K, M = cp.assemble_evp_matrices()

    K_path, M_path = _matrix_paths(out_dir, m, Le)
    K_path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(K_path, K)
    sp.save_npz(M_path, M)
    return K_path, M_path, K.shape[0]


# ---------------------------------------------------------------------------
# Per-target solving
# ---------------------------------------------------------------------------

def _scan_targets_on_rank(
    K: sp.csr_matrix, M: sp.csr_matrix,
    targets: np.ndarray, gamma_target: float, k: int,
    rank: int, size: int,
    log_prefix: str = "",
) -> List[Tuple[float, np.ndarray]]:
    """Solve for k eigenvalues at each target assigned to this rank.

    Targets are divided round-robin: rank handles targets[rank::size].
    Returns a list of (omega_target, eigenvalues) for this rank's slice.

    A failed factorization at a given target is logged and skipped
    (matching the Dedalus script's try/except around solve_sparse).
    """
    from spdee.backends.fd2.compiled_problem import CompiledProblem

    my_targets = targets[rank::size]
    results = []
    t_start = time.time()
    for idx, omega_t in enumerate(my_targets):
        # σ = γ_t − i·ω_t   (spdee λ = −iω convention)
        sigma = complex(gamma_target, -float(omega_t))
        try:
            evals = CompiledProblem.solve_evp_from_matrices(K, M, sigma, k=k)
        except RuntimeError as e:
            print(f"{log_prefix}[rank {rank}] target ω={omega_t:.5f} "
                  f"FAILED: {e}", flush=True)
            continue
        results.append((float(omega_t), evals))
        if idx % 50 == 0 and idx > 0:
            elapsed = time.time() - t_start
            rate = idx / elapsed
            remaining = (len(my_targets) - idx) / max(rate, 1e-9)
            print(f"{log_prefix}[rank {rank}] {idx}/{len(my_targets)} targets, "
                  f"{rate:.2f}/s, ~{remaining/60:.1f} min left", flush=True)
    return results


def _dedup_spectrum(
    all_results: List[Tuple[float, np.ndarray]],
    tol: float = 1e-6,
) -> np.ndarray:
    """Collapse the union of per-target eigenvalues into a unique set.

    Adjacent targets find many of the same eigenvalues; we dedup by
    rounding to `tol` in the complex plane. Returns a sorted array of
    unique complex eigenvalues. NaNs (infinite-eigenvalue artifacts)
    are dropped.
    """
    pool = []
    for _, evals in all_results:
        for ev in np.atleast_1d(evals):
            if np.isfinite(ev):
                pool.append(complex(ev))
    if not pool:
        return np.array([], dtype=complex)
    pool = np.array(pool, dtype=complex)
    # Bucket by rounding real and imaginary parts to multiples of tol.
    keys = (np.round(pool.real / tol).astype(np.int64),
            np.round(pool.imag / tol).astype(np.int64))
    seen = {}
    for ev, kr, ki in zip(pool, keys[0], keys[1]):
        seen.setdefault((int(kr), int(ki)), ev)
    uniq = np.array(list(seen.values()), dtype=complex)
    # Sort by Im (i.e. by ω = −Im) for a tidy spectrum listing.
    return uniq[np.argsort(-uniq.imag)]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bekki + B₀ spectrum scan (Figure-2 style).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/spectrum"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/solar"))
    parser.add_argument("--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES)
    parser.add_argument("--Le-values", type=float, nargs="+", default=DEFAULT_LE_VALUES)
    parser.add_argument("--omega-min", type=float, default=DEFAULT_OMEGA_MIN)
    parser.add_argument("--omega-max", type=float, default=DEFAULT_OMEGA_MAX)
    parser.add_argument("--n-targets", type=int, default=DEFAULT_N_TARGETS)
    parser.add_argument("--gamma-target", type=float, default=DEFAULT_GAMMA_TARGET,
                        help="Fixed damping (Re σ) on each shift. Default -0.001.")
    parser.add_argument("--k", type=int, default=DEFAULT_K,
                        help="Eigenvalues per target. Default 64.")
    parser.add_argument("--Ek", type=float, default=DEFAULT_EK)
    parser.add_argument("--Pm", type=float, default=DEFAULT_PM)
    parser.add_argument("--amplitude-ratio", type=float, default=DEFAULT_AMP_RATIO)
    parser.add_argument("--r-inner", type=float, default=DEFAULT_R_INNER)
    parser.add_argument("--r-outer", type=float, default=DEFAULT_R_OUTER)
    parser.add_argument("--N-r", type=int, default=DEFAULT_N_R)
    parser.add_argument("--N-theta", type=int, default=DEFAULT_N_T)
    parser.add_argument("--order-r", type=int, default=DEFAULT_ORDER_R)
    parser.add_argument("--order-theta", type=int, default=DEFAULT_ORDER_T)
    parser.add_argument("--dedup-tol", type=float, default=1e-6)
    parser.add_argument("--keep-matrices", action="store_true",
                        help="Don't delete the on-disk (K, M) after scanning.")
    parser.add_argument("--first-light", action="store_true",
                        help="1 m, 1 Le, coarse grid, N=48² — serial smoke test.")
    args = parser.parse_args()

    comm, rank, size = _get_mpi()

    if args.first_light:
        args.m_values = [3]
        args.Le_values = [0.0]
        args.n_targets = 101
        args.k = 16
        args.N_r = 48
        args.N_theta = 48
        args.Ek = 3e-3

    targets = np.linspace(args.omega_min, args.omega_max, args.n_targets)

    out_dir = args.output_dir
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        print("=== Bekki+B₀ spectrum scan ===", flush=True)
        print(f"  m values  : {args.m_values}", flush=True)
        print(f"  Le values : {args.Le_values}", flush=True)
        print(f"  ω band    : [{args.omega_min}, {args.omega_max}], "
              f"{args.n_targets} targets", flush=True)
        print(f"  γ target  : {args.gamma_target}", flush=True)
        print(f"  k/target  : {args.k}", flush=True)
        print(f"  grid      : {args.N_r}×{args.N_theta}", flush=True)
        print(f"  MPI ranks : {size}", flush=True)
    comm.Barrier()

    total_start = time.time()

    for m in args.m_values:
        for Le in args.Le_values:
            label = f"m{m:02d}_Le{Le:.4e}"
            log_prefix = f"[{label}] "

            # --- Assemble (K, M) on rank 0, save to disk, then all ranks load.
            K_path, M_path = _matrix_paths(out_dir, m, Le)
            if rank == 0:
                t0 = time.time()
                _assemble_and_save(
                    m, Le, out_dir, args.Ek, args.Pm, args.amplitude_ratio,
                    args.r_inner, args.r_outer, args.N_r, args.N_theta,
                    args.order_r, args.order_theta, str(args.data_dir),
                )
                print(f"{log_prefix}assembled (K, M) in {time.time()-t0:.1f}s, "
                      f"saved to disk", flush=True)
            comm.Barrier()

            # All ranks load the shared matrices.
            K = sp.load_npz(K_path)
            M = sp.load_npz(M_path)

            # --- Scan this rank's slice of targets.
            t0 = time.time()
            rank_results = _scan_targets_on_rank(
                K, M, targets, args.gamma_target, args.k,
                rank, size, log_prefix=log_prefix,
            )
            print(f"{log_prefix}[rank {rank}] done {len(rank_results)} targets "
                  f"in {time.time()-t0:.1f}s", flush=True)

            # --- Gather all ranks' results to rank 0.
            if size > 1:
                gathered = comm.gather(rank_results, root=0)
            else:
                gathered = [rank_results]

            if rank == 0:
                all_results = [item for sub in gathered for item in sub]
                # Save the raw per-target eigenvalues AND the deduped spectrum.
                cell_dir = out_dir / label
                cell_dir.mkdir(parents=True, exist_ok=True)

                # Raw: a 2D ragged save as an object array of (ω_t, evals).
                omega_targets = np.array([r[0] for r in all_results])
                # Stack eigenvalues into a (n_targets, k) array, padding
                # short rows with NaN.
                max_k = max((len(r[1]) for r in all_results), default=0)
                raw = np.full((len(all_results), max_k), np.nan, dtype=complex)
                for i, (_, evals) in enumerate(all_results):
                    raw[i, :len(evals)] = evals
                np.savez(
                    cell_dir / "raw_scan.npz",
                    omega_targets=omega_targets,
                    eigenvalues=raw,
                    m=np.array(m), Le=np.array(Le),
                    Ek=np.array(args.Ek), Pm=np.array(args.Pm),
                    gamma_target=np.array(args.gamma_target),
                    k=np.array(args.k),
                )

                # Deduped unique spectrum.
                uniq = _dedup_spectrum(all_results, tol=args.dedup_tol)
                np.savez(
                    cell_dir / "spectrum.npz",
                    eigenvalues=uniq,
                    omega=-uniq.imag, gamma=-uniq.real,
                    m=np.array(m), Le=np.array(Le),
                    Ek=np.array(args.Ek), Pm=np.array(args.Pm),
                )
                print(f"{log_prefix}gathered {len(all_results)} target-solves "
                      f"→ {len(uniq)} unique modes; saved to {cell_dir}",
                      flush=True)

                # Optionally clean up the big matrix files.
                if not args.keep_matrices:
                    K_path.unlink(missing_ok=True)
                    M_path.unlink(missing_ok=True)
            comm.Barrier()

    if rank == 0:
        print(f"\n=== Scan done in {(time.time()-total_start)/3600:.2f} hr ===",
              flush=True)
        # Clean up the (possibly empty) _matrices dir.
        mat_dir = out_dir / "_matrices"
        if not args.keep_matrices and mat_dir.exists():
            try:
                mat_dir.rmdir()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
