#!/usr/bin/env python3
"""sweep_solar_ball.py — Production-scale sweep of the solar-eigenproblem Step 4 on the
full ball, model S backgrounds, HMI differential rotation.

Physics:
    The compressible convective Step 4 of Bekki et al. (2022, §2.3) with full viscous
    force (including grad(ln ρ₀) · strain correction), thermal
    diffusion (Pr=1), and Neumann entropy BC at the outer surface.
    Ω(r, θ) = HMI fit with uniform_core_blend (rigid Ω_core through
    the radiative interior, full helioseismic profile in the CZ).
    Geometry: ball r ∈ [0, 0.985 R☉]. Origin regularity at r=0;
    impenetrable + stress-free + Neumann entropy at r=0.985.

Sweep:
    m ∈ {1, 2, ..., 16}    (17 azimuthal orders)
    N_r ∈ {256, 512, 1024}
    N_θ ∈ {128, 256, 512}
    Ek = 1e-4 (fixed)

For each (m, N_r, N_θ) the script saves the 30 eigenvalues nearest
the n=0 sectoral Rossby target ω = -2/(m+1), plus solver metadata
(timing, memory, ARPACK iteration count).

Parallelism:
    Each (N_r, N_θ) cell is processed in series (sharing memory across
    a single matrix build is cheap relative to the EVP cost). The 16
    m values for that cell are dispatched in parallel via a
    ProcessPoolExecutor with --jobs workers. Each worker re-imports
    the modules and builds its own (A, B) for that (m, N_r, N_θ).

Resilience:
    Per-cell .npz output to --output-dir. Cells already done are
    skipped on restart. Per-cell exceptions are caught, logged to
    a status file, and the rest of the sweep continues.

Usage:
    python scripts/sweep_solar_ball.py \\
        --output-dir results/ball_Ek1e-4 \\
        --jobs 8 \\
        [--cells 256x128 512x256 ...]   # restrict to specific cells
        [--m-values 1 2 3]               # restrict to specific m
        [--first-light]                  # tiny cell only, for testing
        [--dry-run]                      # print plan, no compute

Outputs (per cell):
    <output_dir>/m{mm:02d}_Nr{Nr}_Nt{Nt}.npz with arrays:
        eigenvalues (complex, shape (k,))
        ω, γ        (real, shape (k,))   (ω = +Im(λ), γ = -Re(λ); ω = −iλ)
        target_omega (scalar)
        k, sigma_imag (scalar)
        N_r, N_theta, m, Ek (scalar metadata)
        solve_time, compile_time (seconds)
        rss_peak (bytes)
    <output_dir>/manifest.json   — aggregate index of all cells done
    <output_dir>/sweep.log       — per-cell progress log
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import resource
import sys
import threading
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Path setup — script lives in scripts/, code lives in spdee/.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_EK = 1e-4
DEFAULT_R_OUTER = 0.985
DEFAULT_K = 30
DEFAULT_M_VALUES = list(range(1, 17))         # m = 1..16
DEFAULT_NR_VALUES = [256, 512, 1024]
DEFAULT_NT_VALUES = [128, 256, 512]
DEFAULT_ORDER_R = 4
DEFAULT_ORDER_THETA = 4
# Heartbeat: workers emit a progress ping during long stages every
# this many seconds, so the driver can confirm liveness even when
# the worker is blocked inside a single SciPy call.
DEFAULT_HEARTBEAT_SECONDS = 30.0
# Per-worker thread budget. SciPy/NumPy load several thread pools
# (OpenMP for SuperLU, OpenBLAS or MKL for dense BLAS) and each may take
# every core, so `--jobs N` would oversubscribe N-fold; 2 threads per
# worker is where SuperLU's long sequential dependency chains stop
# gaining.
DEFAULT_THREADS_PER_WORKER = 2

# Env vars that must be set before numpy/scipy are imported to constrain
# the thread pools; every variable is set because each library reads its
# own.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _apply_thread_limit_env(n_threads: int) -> None:
    """Set all the thread-count env vars to ``n_threads``.

    These vars only constrain library thread pools if set BEFORE the
    library is imported. Calling this from a worker's initializer
    is fine on a spawn-context worker (fresh Python process) but
    is a no-op on a forked worker that inherited an already-imported
    numpy. For belt-and-suspenders, see `_apply_threadpoolctl_limit`
    which works post-import.
    """
    s = str(int(n_threads))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = s


def _apply_threadpoolctl_limit(n_threads: int) -> None:
    """Cap loaded thread pools to ``n_threads`` via `threadpoolctl`.

    Unlike env vars, this works AFTER numerical libraries are
    imported — it detects OpenMP, OpenBLAS, MKL, etc. via dlsym
    and caps each via the library's own runtime API. Belt to the
    env-var suspenders so a forked worker doesn't blow past the
    budget.
    """
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    try:
        threadpool_limits(limits=int(n_threads))
    except Exception:
        # Best effort: if threadpoolctl can't find a pool to limit,
        # we still have the env vars as the primary defense.
        pass


# ---------------------------------------------------------------------------
# Logging: driver-side QueueListener + worker-side QueueHandler
# ---------------------------------------------------------------------------
# Each worker process gets a `multiprocessing.Queue` connected to a
# `logging.handlers.QueueListener` running in the driver. The listener
# forwards each LogRecord to the driver's main handlers (stdout +
# sweep.log), so all worker output ends up in the same place without
# interleaving.
#
# Inside a worker, we attach a `QueueHandler` to the root logger once
# and from then on the worker just uses standard `log.info(...)` calls
# — message formatting, level filtering, and exception capture all use
# the stdlib pipeline.

_WORKER_LOGGER_NAME = "solar_ball_sweep.worker"


def _worker_init(queue, n_threads: int) -> None:
    """Worker process initializer: thread limits + logging.

    Called once per worker process via the ProcessPoolExecutor
    `initializer` hook. Two responsibilities:

    1. Set ``OMP_NUM_THREADS`` / ``OPENBLAS_NUM_THREADS`` /
       ``MKL_NUM_THREADS`` / ... to the chosen budget. With a
       'spawn' context this happens before numpy/scipy are loaded
       in the worker and therefore actually constrains the
       library thread pools.
    2. As a belt-and-suspenders, call ``threadpoolctl.threadpool_limits``
       to constrain any pools that may already have been loaded
       (e.g. by import of `numpy` at module-level above).
    3. Install a ``QueueHandler`` on the root logger so subsequent
       worker `log.info(...)` calls auto-forward to the driver.
    """
    # 1. Env vars — only effective for libraries not yet imported.
    _apply_thread_limit_env(n_threads)
    # 2. Runtime cap on already-loaded pools.
    _apply_threadpoolctl_limit(n_threads)
    # 3. Logging.
    from logging.handlers import QueueHandler
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(QueueHandler(queue))
    root.setLevel(logging.INFO)


def _worker_log(m: int, stage: str, msg: str = "") -> None:
    """Emit a progress line from inside a worker.

    Routes through the worker's logger, which the `QueueHandler`
    forwards to the driver. The driver's formatter prepends a
    timestamp; we add the (m, stage) prefix here so the field
    alignment is consistent across all worker messages.
    """
    log = logging.getLogger(_WORKER_LOGGER_NAME)
    if msg:
        log.info(f"m={m:>2} [{stage:<11}] {msg}")
    else:
        log.info(f"m={m:>2} [{stage:<11}]")


class _Heartbeat:
    """Context manager: log "started", periodic "still running",
    and "done" / "FAILED" lines around a long-running stage.

    The intervening worker logger does the heavy lifting; this class
    is just a thin scheduler so the user sees a heartbeat while
    SciPy is blocked inside a single call.

    Usage:
        with _Heartbeat(m, "solve", interval=30.0):
            cp.solve_evp(...)
    """
    def __init__(self, m: int, stage: str, interval: float = 30.0):
        self._m = m
        self._stage = stage
        self._interval = float(interval)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t_started = 0.0

    def __enter__(self):
        self._t_started = time.time()
        _worker_log(self._m, self._stage, "started")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        elapsed = time.time() - self._t_started
        if exc_type is None:
            _worker_log(self._m, self._stage, f"done ({elapsed:.1f}s)")
        else:
            _worker_log(self._m, self._stage,
                        f"FAILED after {elapsed:.1f}s: {exc_type.__name__}")
        return False  # don't suppress exceptions

    def _loop(self):
        next_at = self._t_started + self._interval
        while not self._stop.is_set():
            now = time.time()
            wait = max(0.05, min(1.0, next_at - now))
            if self._stop.wait(timeout=wait):
                return
            now = time.time()
            if now >= next_at:
                elapsed = now - self._t_started
                _worker_log(self._m, self._stage,
                            f"still running ({elapsed:.1f}s)")
                next_at = now + self._interval


# ---------------------------------------------------------------------------
# Per-(m, N_r, N_θ) worker
# ---------------------------------------------------------------------------

def _solve_one_cell(
    m: int,
    N_r: int,
    N_theta: int,
    Ek: float,
    r_outer: float,
    k: int,
    order_r: int,
    order_theta: int,
    data_dir: str,
    output_path: str,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
) -> dict:
    """Run a single (m, N_r, N_θ) cell, save .npz, return a status dict.

    This is the per-process entry point — designed to be top-level
    so ProcessPoolExecutor can pickle it. Worker-side progress is
    emitted via the stdlib logger and routed to the driver through
    the `QueueHandler` installed in `_worker_init`.
    """
    import numpy as np
    from pathlib import Path

    _worker_log(m, "started",
                f"N_r={N_r}, N_θ={N_theta}, Ek={Ek:.1e}")

    # Diagnostic: report the actual thread pool sizes observed
    # AFTER numpy/scipy are loaded. If the cap didn't take effect
    # (e.g. fork context, threadpoolctl missing), the user sees it
    # here instead of discovering it later via htop.
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        pool_summary = ", ".join(
            f"{p.get('prefix', p.get('user_api', '?'))}:{p['num_threads']}"
            for p in pools
        ) if pools else "(no pools detected)"
        _worker_log(m, "threads", pool_summary)
    except ImportError:
        # threadpoolctl missing — fall back to reporting the env
        # vars we set. The env vars do work (with spawn context) but
        # we can't directly verify they're being honored. Suggest
        # `pip install threadpoolctl` for the verification layer.
        env_summary = ", ".join(
            f"{var.replace('_NUM_THREADS', '')}={os.environ.get(var, '?')}"
            for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS")
        )
        _worker_log(m, "threads",
                    f"env={env_summary}  "
                    f"(pip install threadpoolctl for runtime verification)")

    # Re-import inside the worker (each process has its own state).
    _worker_log(m, "imports", "loading spdee modules")
    from spdee.backgrounds.solar import SolarBackgrounds
    from spdee.backends.fd2.grid import make_ball_grid
    from spdee.backends.fd2.compiled_problem import compile_problem
    from test_solar_shell import TestCompressibleStep4

    t_start = time.time()

    _worker_log(m, "backgrounds", "loading Model S + HMI")
    solar = SolarBackgrounds(
        Path(data_dir), ('ball', r_outer),
        nondimensional=True, uniform_core_blend=True,
    )

    _worker_log(m, "system", "building Step 4 (ball, Neumann entropy)")
    sys_ = TestCompressibleStep4._build(
        solar, m_val=m, Ek=Ek,
        entropy_bc="neumann", geometry="ball",
    )
    _worker_log(m, "system",
                f"{len(sys_.equations)} eqs, "
                f"{len(sys_.boundary_conditions)} BCs")

    _worker_log(m, "grid",
                f"ball N_r×N_θ={N_r}×{N_theta}, "
                f"r∈[0, {r_outer}], DOF={7*N_r*N_theta:,}")
    grid = make_ball_grid(N_r=N_r, N_theta=N_theta, r_outer=r_outer)

    t_compile = time.time()
    with _Heartbeat(m, "compile", heartbeat_seconds):
        cp = compile_problem(
            sys_, grid, m=m, order_r=order_r, order_theta=order_theta,
        )
    compile_time = time.time() - t_compile
    # Report matrix sparsity numbers — useful for sizing parallelism.
    A_nnz = int(cp.matrices.A_total.nnz)
    B_nnz = int(cp.matrices.B_total.nnz)
    state_size = int(cp.matrices.A_total.shape[0])
    _worker_log(m, "compile",
                f"K: {state_size:,}² with {A_nnz:,} nnz "
                f"({A_nnz / max(state_size, 1):.1f}/row); "
                f"M: {B_nnz:,} nnz")

    target = -2.0 / (m + 1)
    sigma = 1j * target           # λ = +iω (ω = −iλ); physical mode at λ≈+iω

    # Enable phase profiling for this solve so we can report a
    # breakdown of K build / M build / shifted-matrix / LU /
    # ARPACK times. The profiling module accumulates into
    # process-global counters; reset before each call so we read
    # back times for THIS solve only.
    from spdee.backends.fd2 import profiling as _prof
    _prof.enable()
    _prof.reset()

    t_solve = time.time()
    _worker_log(m, "solve",
                f"k={k}, σ=+i·({target:+.4f}); "
                f"LU + ARPACK shift-invert")
    # The solve itself is a single SciPy call from our vantage point;
    # heartbeat tells the user it's alive.
    with _Heartbeat(m, "solve", heartbeat_seconds):
        res = cp.solve_evp(k=k, sigma=sigma)
    solve_time = time.time() - t_solve

    # Pull out the recorded sub-stage timings.
    prof_snap = _prof.snapshot()
    t_build_K   = prof_snap.get("solve_evp/build_K",   (0.0, 0))[0]
    t_build_M   = prof_snap.get("solve_evp/build_M",   (0.0, 0))[0]
    t_shifted   = prof_snap.get("solve_evp/shifted_matrix", (0.0, 0))[0]
    t_lu        = prof_snap.get("solve_evp/lu_decompose", (0.0, 0))[0]
    t_arpack    = prof_snap.get("solve_evp/arpack",    (0.0, 0))[0]
    t_matvec_total, n_matvec = prof_snap.get(
        "solve_evp/arpack_matvec", (0.0, 0),
    )
    t_unpack    = prof_snap.get("solve_evp/unpack",    (0.0, 0))[0]
    _worker_log(m, "timings",
                f"K={t_build_K:.1f}s  M={t_build_M:.1f}s  "
                f"(K−σM)={t_shifted:.1f}s  LU={t_lu:.1f}s  "
                f"ARPACK={t_arpack:.1f}s "
                f"({n_matvec} matvec ×{t_matvec_total/max(n_matvec,1)*1000:.1f}ms = "
                f"{t_matvec_total:.1f}s)")

    eigs = np.array(res.eigenvalues)
    # Canonical convention ω = −iλ (see EvpResult.eigenvalues_to_omega):
    #   Re(ω) = +Im(λ),  Im(ω) = γ = −Re(λ)  (γ < 0 ⇒ damped).
    omega = eigs.imag     # Re(ω) = +Im(λ)
    gamma = -eigs.real    # Im(ω) = γ = −Re(λ)

    # Sort by proximity to the target shift (ARPACK doesn't guarantee
    # ordering, and we want the user to find the Rossby trajectory
    # easily later).
    sort_idx = np.argsort(np.abs(eigs - sigma))
    eigs = eigs[sort_idx]
    omega = omega[sort_idx]
    gamma = gamma[sort_idx]
    n_finite = int(np.sum(np.isfinite(eigs)))
    _worker_log(m, "solve",
                f"got {len(eigs)} eigenvalues "
                f"({n_finite} finite, {len(eigs) - n_finite} ∞/NaN); "
                f"top: ω={omega[0]:+.5f}, γ={gamma[0]:+.5f}")

    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _worker_log(m, "save", f"writing {out_path.name}")
    np.savez(
        out_path,
        eigenvalues=eigs,
        omega=omega,
        gamma=gamma,
        target_omega=np.array(target, dtype=float),
        k=np.array(k, dtype=int),
        sigma_imag=np.array(sigma.imag, dtype=float),
        m=np.array(m, dtype=int),
        N_r=np.array(N_r, dtype=int),
        N_theta=np.array(N_theta, dtype=int),
        Ek=np.array(Ek, dtype=float),
        r_outer=np.array(r_outer, dtype=float),
        order_r=np.array(order_r, dtype=int),
        order_theta=np.array(order_theta, dtype=int),
        compile_time=np.array(compile_time, dtype=float),
        solve_time=np.array(solve_time, dtype=float),
        rss_peak_kb=np.array(rss_peak, dtype=int),
        # Sub-stage timings from the profiling snapshot.
        t_build_K=np.array(t_build_K, dtype=float),
        t_build_M=np.array(t_build_M, dtype=float),
        t_shifted_matrix=np.array(t_shifted, dtype=float),
        t_lu_decompose=np.array(t_lu, dtype=float),
        t_arpack=np.array(t_arpack, dtype=float),
        t_arpack_matvec_total=np.array(t_matvec_total, dtype=float),
        n_arpack_matvec=np.array(n_matvec, dtype=int),
        t_unpack=np.array(t_unpack, dtype=float),
    )

    elapsed = time.time() - t_start
    _worker_log(m, "done",
                f"elapsed={elapsed:.1f}s, RSS peak={rss_peak/1024/1024:.2f} GB")
    return {
        "ok": True,
        "m": m,
        "N_r": N_r,
        "N_theta": N_theta,
        "Ek": Ek,
        "compile_time": compile_time,
        "solve_time": solve_time,
        "elapsed": elapsed,
        "rss_peak_kb": rss_peak,
        "output_path": str(out_path),
        # Headline numbers for the log: closest mode (within the
        # Rossby band) and the smallest two |λ - sigma|.
        "closest_omegas": omega[:5].tolist(),
        "closest_gammas": gamma[:5].tolist(),
        # Sub-stage timings (seconds).
        "t_build_K": t_build_K,
        "t_build_M": t_build_M,
        "t_shifted_matrix": t_shifted,
        "t_lu_decompose": t_lu,
        "t_arpack": t_arpack,
        "t_arpack_matvec_total": t_matvec_total,
        "n_arpack_matvec": int(n_matvec),
        "t_unpack": t_unpack,
    }


def _solve_one_cell_safe(*args, **kwargs) -> dict:
    """Wrapper that catches exceptions and returns a status dict."""
    try:
        return _solve_one_cell(*args, **kwargs)
    except Exception as e:
        return {
            "ok": False,
            "error_type": type(e).__name__,
            "error_msg": str(e),
            "traceback": traceback.format_exc(),
            "args": args,
            "kwargs": kwargs,
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> Tuple[logging.Logger, List[logging.Handler]]:
    """Configure the driver-side root logger.

    Returns (driver_logger, handlers) so callers can also pass the
    handlers to a `QueueListener` for worker-side messages.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Attach to root so worker records forwarded via QueueListener
    # also land in these handlers.
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    root.addHandler(sh)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    root.addHandler(fh)
    driver_logger = logging.getLogger("solar_ball_sweep")
    driver_logger.setLevel(logging.INFO)
    return driver_logger, [sh, fh]


def _cell_output_path(out_dir: Path, m: int, N_r: int, N_theta: int) -> Path:
    return out_dir / f"m{m:02d}_Nr{N_r:04d}_Nt{N_theta:04d}.npz"


def _cell_done_and_valid(path: Path) -> bool:
    """Cell is 'done' if the .npz exists, has positive size, and can
    be opened. Use this to skip on restart while detecting truncated
    files from a previous crash."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path) as data:
            # Touch the required key.
            _ = data["eigenvalues"]
        return True
    except Exception:
        return False


def _parse_cells_arg(arg_list: Optional[List[str]]) -> Optional[List[Tuple[int, int]]]:
    """Parse --cells like '256x128 512x256' into [(256,128), (512,256)]."""
    if arg_list is None:
        return None
    cells = []
    for spec in arg_list:
        try:
            a, b = spec.lower().split("x")
            cells.append((int(a), int(b)))
        except Exception:
            raise SystemExit(f"--cells: bad spec {spec!r}, expected like '256x128'")
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", required=True, type=Path,
        help="Where per-cell .npz, manifest.json, and sweep.log are written.",
    )
    parser.add_argument(
        "--jobs", type=int, default=4,
        help="Number of parallel m-solves per (N_r, N_θ) cell. Default 4.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=_REPO_ROOT / "data" / "solar",
        help="Path to the solar/ data directory.",
    )
    parser.add_argument(
        "--Ek", type=float, default=DEFAULT_EK,
        help=f"Ekman number. Default {DEFAULT_EK:.0e}.",
    )
    parser.add_argument(
        "--r-outer", type=float, default=DEFAULT_R_OUTER,
        help=f"Outer radial boundary in R_sun. Default {DEFAULT_R_OUTER}.",
    )
    parser.add_argument(
        "--k", type=int, default=DEFAULT_K,
        help=f"Number of eigenvalues per (m, cell). Default {DEFAULT_K}.",
    )
    parser.add_argument(
        "--m-values", nargs="+", type=int, default=DEFAULT_M_VALUES,
        help="Azimuthal orders to sweep. Default %(default)s.",
    )
    parser.add_argument(
        "--Nr-values", nargs="+", type=int, default=DEFAULT_NR_VALUES,
        help="Radial resolutions to sweep. Default %(default)s.",
    )
    parser.add_argument(
        "--Nt-values", nargs="+", type=int, default=DEFAULT_NT_VALUES,
        help="Latitudinal resolutions to sweep. Default %(default)s.",
    )
    parser.add_argument(
        "--cells", nargs="+", default=None,
        metavar="NrxNt",
        help="Restrict to specific (N_r, N_θ) cells, e.g. 256x128. "
             "If given, --Nr-values and --Nt-values are ignored.",
    )
    parser.add_argument(
        "--order-r", type=int, default=DEFAULT_ORDER_R,
        help="Radial FD order. Default %(default)d.",
    )
    parser.add_argument(
        "--order-theta", type=int, default=DEFAULT_ORDER_THETA,
        help="Latitudinal FD order. Default %(default)d.",
    )
    parser.add_argument(
        "--heartbeat", type=float, default=DEFAULT_HEARTBEAT_SECONDS,
        help=f"Seconds between 'still running' pings from each "
             f"worker during long stages (compile, solve). "
             f"Default {DEFAULT_HEARTBEAT_SECONDS}s.",
    )
    parser.add_argument(
        "--threads-per-worker", type=int,
        default=DEFAULT_THREADS_PER_WORKER,
        metavar="N",
        help=f"Cap each worker's BLAS/LAPACK/OMP thread pool to N "
             f"threads. SciPy's splu gets diminishing returns "
             f"beyond 2 threads, so the default ({DEFAULT_THREADS_PER_WORKER}) "
             f"keeps total core usage at --jobs × N. Set higher (4–8) "
             f"if you have many cores and few jobs and want each "
             f"individual solve to run faster.",
    )
    parser.add_argument(
        "--first-light", action="store_true",
        help="Single cheap cell (m=3, N_r=128, N_θ=64) only, for "
             "smoke testing. Overrides all sweep arguments.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned cells and exit without computing.",
    )
    args = parser.parse_args()

    # Resolve sweep dimensions.
    if args.first_light:
        m_values = [3]
        cells_list = [(128, 64)]
    elif args.cells:
        m_values = list(args.m_values)
        cells_list = _parse_cells_arg(args.cells)
    else:
        m_values = list(args.m_values)
        cells_list = [(nr, nt) for nr in args.Nr_values for nt in args.Nt_values]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "sweep.log"
    log, log_handlers = _setup_logger(log_path)

    # Multiprocessing logging: each worker writes via a QueueHandler
    # connected to this queue; the driver-side QueueListener forwards
    # records to the same handlers as the driver's own log calls.
    # Create the queue from the SAME 'spawn' context the executor
    # will use, otherwise the internal SemLock can't be shared
    # across contexts.
    mp_ctx = mp.get_context("spawn")
    from logging.handlers import QueueListener
    log_queue = mp_ctx.Queue(-1)
    log_listener = QueueListener(log_queue, *log_handlers, respect_handler_level=True)
    log_listener.start()

    log.info("=" * 78)
    log.info("Solar-eigenproblem Step 4 ball sweep")
    log.info("=" * 78)
    log.info(f"output_dir       : {args.output_dir}")
    log.info(f"jobs (parallel m): {args.jobs}")
    log.info(f"threads/worker   : {args.threads_per_worker} "
             f"(total cores in use ≤ {args.jobs * args.threads_per_worker})")
    log.info(f"Ek               : {args.Ek:.3e}")
    log.info(f"r_outer          : {args.r_outer}")
    log.info(f"k (per solve)    : {args.k}")
    log.info(f"order_r, order_θ : {args.order_r}, {args.order_theta}")
    log.info(f"heartbeat        : {args.heartbeat}s")
    log.info(f"m values         : {m_values}")
    log.info(f"cells (N_r×N_θ)  : {cells_list}")
    log.info(f"total cells      : {len(cells_list)}")
    log.info(f"total m-solves   : {len(cells_list) * len(m_values)}")

    if args.dry_run:
        log.info("--dry-run: exiting without compute.")
        log_listener.stop()
        return 0

    # ----- Iterate cells -----
    manifest_path = args.output_dir / "manifest.json"
    manifest: dict = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            log.info(f"Existing manifest loaded ({len(manifest)} entries).")
        except Exception as e:
            log.warning(f"Existing manifest unreadable ({e!r}); starting fresh.")
            manifest = {}

    total_start = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    for cell_idx, (N_r, N_theta) in enumerate(cells_list, 1):
        cell_label = f"({N_r}×{N_theta})"
        log.info("")
        log.info(f"=== Cell {cell_idx}/{len(cells_list)}: N_r={N_r}, N_θ={N_theta} ===")

        # Determine which m values still need work in this cell.
        pending_args = []
        for m in m_values:
            out_path = _cell_output_path(args.output_dir, m, N_r, N_theta)
            if _cell_done_and_valid(out_path):
                log.info(f"  m={m:>2} already done at {out_path.name} — skip")
                n_skipped += 1
                continue
            pending_args.append((m, out_path))

        if not pending_args:
            log.info(f"  Nothing to do in cell {cell_label}.")
            continue

        cell_start = time.time()

        # Dispatch parallel m solves for this cell.
        log.info(f"  Dispatching {len(pending_args)} m-solves "
                 f"in cell {cell_label} (jobs={args.jobs}, "
                 f"threads/worker={args.threads_per_worker})...")
        # Use 'spawn' so each worker is a fresh Python process: the
        # thread-limit env vars set in `_worker_init` then take
        # effect BEFORE numpy/scipy are imported. With the default
        # 'fork' on Linux, the worker inherits an already-imported
        # numpy and the env vars are too late.
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            mp_context=mp_ctx,
            initializer=_worker_init,
            initargs=(log_queue, args.threads_per_worker),
        ) as ex:
            future_to_m = {}
            for m, out_path in pending_args:
                fut = ex.submit(
                    _solve_one_cell_safe,
                    m, N_r, N_theta, args.Ek, args.r_outer, args.k,
                    args.order_r, args.order_theta,
                    str(args.data_dir), str(out_path),
                    args.heartbeat,
                )
                future_to_m[fut] = m

            for fut in as_completed(future_to_m):
                m = future_to_m[fut]
                result = fut.result()
                if result.get("ok"):
                    n_done += 1
                    om_top = result["closest_omegas"][0]
                    ga_top = result["closest_gammas"][0]
                    log.info(
                        f"  m={m:>2} OK  compile={result['compile_time']:6.1f}s "
                        f"solve={result['solve_time']:7.1f}s  "
                        f"rss={result['rss_peak_kb']/1024/1024:5.2f} GB  "
                        f"top: ω={om_top:+.5f}, γ={ga_top:+.5f}"
                    )
                    # Breakdown of where the solve_time went.
                    log.info(
                        f"  m={m:>2}      "
                        f"K={result['t_build_K']:5.1f}s  "
                        f"M={result['t_build_M']:5.1f}s  "
                        f"(K−σM)={result['t_shifted_matrix']:5.1f}s  "
                        f"LU={result['t_lu_decompose']:6.1f}s  "
                        f"ARPACK={result['t_arpack']:6.1f}s "
                        f"({result['n_arpack_matvec']:>4d} matvec, "
                        f"avg "
                        f"{1000*result['t_arpack_matvec_total']/max(result['n_arpack_matvec'],1):.1f} ms)"
                    )
                    manifest[f"m{m:02d}_Nr{N_r}_Nt{N_theta}"] = {
                        "m": m, "N_r": N_r, "N_theta": N_theta,
                        "Ek": args.Ek,
                        "compile_time": result["compile_time"],
                        "solve_time": result["solve_time"],
                        "rss_peak_kb": result["rss_peak_kb"],
                        "output": Path(result["output_path"]).name,
                        "top_omega": om_top,
                        "top_gamma": ga_top,
                        "t_build_K": result["t_build_K"],
                        "t_build_M": result["t_build_M"],
                        "t_shifted_matrix": result["t_shifted_matrix"],
                        "t_lu_decompose": result["t_lu_decompose"],
                        "t_arpack": result["t_arpack"],
                        "n_arpack_matvec": result["n_arpack_matvec"],
                    }
                else:
                    n_failed += 1
                    log.error(
                        f"  m={m:>2} FAILED: {result['error_type']}: "
                        f"{result['error_msg']}"
                    )
                    log.error(
                        "      traceback (head):\n      " +
                        result["traceback"].splitlines()[-1]
                    )
                    manifest[f"m{m:02d}_Nr{N_r}_Nt{N_theta}"] = {
                        "m": m, "N_r": N_r, "N_theta": N_theta,
                        "Ek": args.Ek,
                        "error": f"{result['error_type']}: {result['error_msg']}",
                    }

        # Persist manifest after each cell.
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        cell_elapsed = time.time() - cell_start
        log.info(
            f"  Cell {cell_label} done in {cell_elapsed/60:.1f} min "
            f"({len(pending_args)} solves, "
            f"{cell_elapsed/max(len(pending_args), 1):.1f} s/solve avg)."
        )

    total_elapsed = time.time() - total_start
    log.info("")
    log.info("=" * 78)
    log.info(
        f"Sweep done in {total_elapsed/3600:.2f} hr. "
        f"new={n_done}, skipped={n_skipped}, failed={n_failed}."
    )
    log.info(f"Manifest: {manifest_path}")
    log.info(f"Log:      {log_path}")
    log.info("=" * 78)
    log_listener.stop()
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
