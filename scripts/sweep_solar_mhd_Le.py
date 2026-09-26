#!/usr/bin/env python3
"""Solar eigenproblem + dipolar B₀ : Lehnert-number sweep at fixed m, fixed grid.

Studies how the dominant n=0 sectoral Rossby mode shifts as the
background magnetic-field amplitude is increased. The field is
divergence-free, dipolar in θ-structure, with an exponential radial
profile that maps 30 kG at the tachocline to 1 G at the surface (a
factor of 30000). The Lehnert number Le = v_A^(ref) / (Ω₀ R) controls
the overall amplitude.

Usage
-----

  # Default sweep: 9 Le values from 0 to 0.1, m=3, Ek=1e-4, N_r×N_θ = 256×128.
  python scripts/sweep_solar_mhd_Le.py --output-dir results/Le_sweep

  # Multi-m sweep with finer Le grid.
  python scripts/sweep_solar_mhd_Le.py \
      --output-dir results/Le_sweep \
      --m-values 1 2 3 4 5 \
      --Le-values 0 0.001 0.01 0.03 0.05 0.1 \
      --N-r 256 --N-theta 128 --jobs 16

  # First-light: single cheap solve.
  python scripts/sweep_solar_mhd_Le.py --first-light

Output
------

Each (m, Le) cell produces an .npz file containing the eigenvalues
(sorted by proximity to the target Rossby frequency), timings, and
metadata. A `manifest.json` accumulates the top-eigenvalue summary
across all solves. The `sweep.log` captures the full progress feed.

Hydro recovery: with Le=0, the spectrum bit-exactly contains the
pure solar-eigenproblem Step 4 result. The sweep starts at Le=0 by default so
you have an explicit hydro baseline in the output.
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
from typing import List, Tuple

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "tests"))


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_M_VALUES = [3]
DEFAULT_LE_VALUES = [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 5e-2, 7e-2, 1e-1]
DEFAULT_EK = 1e-4
DEFAULT_PM = 1.0                       # magnetic Prandtl
DEFAULT_AMPLITUDE_RATIO = 30000.0      # |B(r_in)/B(r_out)|
DEFAULT_R_INNER = 0.71
DEFAULT_R_OUTER = 0.985
DEFAULT_N_R = 256
DEFAULT_N_T = 128
DEFAULT_K = 30
DEFAULT_ORDER_R = 4
DEFAULT_ORDER_THETA = 4
DEFAULT_HEARTBEAT_SECONDS = 30.0
DEFAULT_THREADS_PER_WORKER = 2

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)


def _apply_thread_limit_env(n: int) -> None:
    s = str(int(n))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = s


def _apply_threadpoolctl_limit(n: int) -> None:
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=int(n))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Logging plumbing (same pattern as sweep_solar_ball.py)
# ---------------------------------------------------------------------------

_WORKER_LOGGER_NAME = "solar_eigenmodes_sweep.worker"


def _worker_init(queue, n_threads: int) -> None:
    _apply_thread_limit_env(n_threads)
    _apply_threadpoolctl_limit(n_threads)
    from logging.handlers import QueueHandler
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(QueueHandler(queue))
    root.setLevel(logging.INFO)


def _worker_log(m: int, Le: float, stage: str, msg: str = "") -> None:
    log = logging.getLogger(_WORKER_LOGGER_NAME)
    tag = f"m={m:>2} Le={Le:.0e}"
    if msg:
        log.info(f"{tag} [{stage:<11}] {msg}")
    else:
        log.info(f"{tag} [{stage:<11}]")


class _Heartbeat:
    def __init__(self, m, Le, stage, interval=30.0):
        self._m = m; self._Le = Le
        self._stage = stage; self._interval = float(interval)
        self._stop = threading.Event()
        self._thread = None
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.time()
        _worker_log(self._m, self._Le, self._stage, "started")
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        elapsed = time.time() - self._t0
        if exc_type is None:
            _worker_log(self._m, self._Le, self._stage, f"done ({elapsed:.1f}s)")
        else:
            _worker_log(self._m, self._Le, self._stage,
                        f"FAILED after {elapsed:.1f}s: {exc_type.__name__}")
        return False

    def _loop(self):
        next_at = self._t0 + self._interval
        while not self._stop.is_set():
            now = time.time()
            wait = max(0.05, min(1.0, next_at - now))
            if self._stop.wait(timeout=wait):
                return
            now = time.time()
            if now >= next_at:
                _worker_log(self._m, self._Le, self._stage,
                            f"still running ({now - self._t0:.1f}s)")
                next_at = now + self._interval


# ---------------------------------------------------------------------------
# The per-cell worker
# ---------------------------------------------------------------------------

def _solve_one_cell(
    m: int, Le: float,
    N_r: int, N_theta: int,
    Ek: float, Pm: float,
    amplitude_ratio: float,
    r_inner: float, r_outer: float,
    k: int, order_r: int, order_theta: int,
    data_dir: str, output_path: str,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
) -> dict:
    import numpy as np
    _worker_log(m, Le, "started",
                f"N={N_r}×{N_theta}, Ek={Ek:.1e}, Pm={Pm}")

    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
        psum = ", ".join(f"{p.get('prefix','?')}:{p['num_threads']}" for p in pools)
        _worker_log(m, Le, "threads", psum or "(no pools)")
    except ImportError:
        env_sum = ", ".join(
            f"{v.replace('_NUM_THREADS','')}={os.environ.get(v,'?')}"
            for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")
        )
        _worker_log(m, Le, "threads", env_sum)

    _worker_log(m, Le, "imports", "loading spdee + solar_eigenmodes")
    from spdee.backgrounds.solar import SolarBackgrounds
    from spdee.backends.fd2.grid import make_shell_grid
    from spdee.backends.fd2.compiled_problem import compile_problem
    from spdee.applications.solar_eigenmodes import make_solar_shell_system
    from spdee.backends.fd2 import profiling as _prof

    t_start = time.time()
    _worker_log(m, Le, "backgrounds", "loading Model S + HMI")
    solar = SolarBackgrounds(
        Path(data_dir), ('shell', r_inner, r_outer),
        nondimensional=True, uniform_core_blend=True,
    )

    _worker_log(m, Le, "system", "building solar eigenproblem + B₀")
    sys_ = make_solar_shell_system(
        solar, m_val=m, Ek=Ek, Le=Le, Pm=Pm,
        amplitude_ratio=amplitude_ratio,
        entropy_bc="neumann", geometry="shell",
        r_inner=r_inner, r_outer=r_outer,
    )
    _worker_log(m, Le, "system",
                f"{len(sys_.equations)} eqs, "
                f"{len(sys_.boundary_conditions)} BCs")

    _worker_log(m, Le, "grid",
                f"N_r×N_θ={N_r}×{N_theta}, DOF={8*N_r*N_theta:,}")
    grid = make_shell_grid(
        N_r=N_r, N_theta=N_theta, r_inner=r_inner, r_outer=r_outer,
    )

    t_compile = time.time()
    with _Heartbeat(m, Le, "compile", heartbeat_seconds):
        cp = compile_problem(
            sys_, grid, m=m, order_r=order_r, order_theta=order_theta,
        )
    compile_time = time.time() - t_compile
    A_nnz = int(cp.matrices.A_total.nnz)
    state_size = int(cp.matrices.A_total.shape[0])
    _worker_log(m, Le, "compile",
                f"K: {state_size:,}² @ {A_nnz:,} nnz "
                f"({A_nnz/max(state_size,1):.1f}/row)")

    # Pick target shift: Rossby asymptote -2/(m+1).
    target = -2.0 / (m + 1)
    sigma = -1j * target

    _prof.enable(); _prof.reset()

    t_solve = time.time()
    _worker_log(m, Le, "solve",
                f"k={k}, σ=-i·({target:+.4f}); LU + ARPACK")
    with _Heartbeat(m, Le, "solve", heartbeat_seconds):
        res = cp.solve_evp(k=k, sigma=sigma)
    solve_time = time.time() - t_solve

    snap = _prof.snapshot()
    t_build_K   = snap.get("solve_evp/build_K",   (0.0, 0))[0]
    t_build_M   = snap.get("solve_evp/build_M",   (0.0, 0))[0]
    t_lu        = snap.get("solve_evp/lu_decompose", (0.0, 0))[0]
    t_arpack    = snap.get("solve_evp/arpack", (0.0, 0))[0]
    t_mv_total, n_mv = snap.get("solve_evp/arpack_matvec", (0.0, 0))
    _worker_log(m, Le, "timings",
                f"K={t_build_K:.1f}s  M={t_build_M:.1f}s  "
                f"LU={t_lu:.1f}s  ARPACK={t_arpack:.1f}s "
                f"({n_mv} mv ×{1000*t_mv_total/max(n_mv,1):.1f}ms)")

    eigs = np.array(res.eigenvalues)
    omega = -eigs.imag
    gamma = -eigs.real
    # Sort by proximity to target shift.
    sort_idx = np.argsort(np.abs(eigs - sigma))
    eigs = eigs[sort_idx]
    omega = omega[sort_idx]
    gamma = gamma[sort_idx]

    # Also identify the dominant Rossby mode: high γ damping
    # (away from purely-magnetic decay modes, which have small γ).
    # Heuristic: γ < -0.05 (sufficient damping), ω in (-0.8, -0.1)
    # (Rossby band). For solar nominal Le ~ 0.01 the shift is tiny,
    # but at Le ~ 0.1 the band may broaden — keep a wide window.
    keep = (gamma < -0.05) & (omega < -0.05) & (omega > -1.0)
    if np.any(keep):
        idx = np.where(keep)[0]
        sub_idx = idx[np.argsort(np.abs(omega[idx] - target))]
        rossby_omega = float(omega[sub_idx[0]])
        rossby_gamma = float(gamma[sub_idx[0]])
    else:
        rossby_omega = float("nan")
        rossby_gamma = float("nan")
    _worker_log(m, Le, "solve",
                f"got {len(eigs)} eigs; closest: ω={omega[0]:+.5f}, γ={gamma[0]:+.5f}; "
                f"Rossby: ω={rossby_omega:+.5f}, γ={rossby_gamma:+.5f}")

    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        eigenvalues=eigs, omega=omega, gamma=gamma,
        target_omega=np.array(target, dtype=float),
        sigma_imag=np.array(sigma.imag, dtype=float),
        m=np.array(m, dtype=int),
        Le=np.array(Le, dtype=float),
        Ek=np.array(Ek, dtype=float),
        Pm=np.array(Pm, dtype=float),
        amplitude_ratio=np.array(amplitude_ratio, dtype=float),
        N_r=np.array(N_r, dtype=int),
        N_theta=np.array(N_theta, dtype=int),
        r_inner=np.array(r_inner, dtype=float),
        r_outer=np.array(r_outer, dtype=float),
        compile_time=np.array(compile_time, dtype=float),
        solve_time=np.array(solve_time, dtype=float),
        rss_peak_kb=np.array(rss_peak, dtype=int),
        t_build_K=np.array(t_build_K, dtype=float),
        t_build_M=np.array(t_build_M, dtype=float),
        t_lu_decompose=np.array(t_lu, dtype=float),
        t_arpack=np.array(t_arpack, dtype=float),
        n_arpack_matvec=np.array(n_mv, dtype=int),
        rossby_omega=np.array(rossby_omega, dtype=float),
        rossby_gamma=np.array(rossby_gamma, dtype=float),
    )

    elapsed = time.time() - t_start
    _worker_log(m, Le, "done",
                f"elapsed={elapsed:.1f}s, RSS peak={rss_peak/1024/1024:.2f} GB")
    return {
        "ok": True, "m": m, "Le": Le,
        "N_r": N_r, "N_theta": N_theta,
        "compile_time": compile_time, "solve_time": solve_time,
        "elapsed": elapsed, "rss_peak_kb": rss_peak,
        "output_path": str(out_path),
        "rossby_omega": rossby_omega,
        "rossby_gamma": rossby_gamma,
        "t_lu_decompose": t_lu, "t_arpack": t_arpack,
        "n_arpack_matvec": int(n_mv),
    }


def _solve_one_cell_safe(*args, **kwargs):
    try:
        return _solve_one_cell(*args, **kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _setup_logger(log_path: Path) -> Tuple[logging.Logger, List[logging.Handler]]:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
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
    driver = logging.getLogger("solar_eigenmodes_sweep")
    driver.setLevel(logging.INFO)
    return driver, [sh, fh]


def _cell_output_path(out_dir: Path, m: int, Le: float) -> Path:
    # Encode Le with enough precision to be unique across reasonable sweeps.
    if Le == 0:
        tag = "Le0"
    else:
        tag = f"Le{Le:.4e}".replace("+", "p").replace("-", "n").replace(".", "_")
    return out_dir / f"m{m:02d}_{tag}.npz"


def _cell_done_and_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path) as d:
            return "eigenvalues" in d.files and len(d["eigenvalues"]) > 0
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Solar eigenproblem + B₀ Le sweep at fixed m, fixed grid.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/Le_sweep"),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/solar"),
    )
    parser.add_argument(
        "--m-values", type=int, nargs="+", default=DEFAULT_M_VALUES,
    )
    parser.add_argument(
        "--Le-values", type=float, nargs="+", default=DEFAULT_LE_VALUES,
        help=f"Lehnert numbers. Default: {DEFAULT_LE_VALUES}",
    )
    parser.add_argument("--Ek", type=float, default=DEFAULT_EK)
    parser.add_argument("--Pm", type=float, default=DEFAULT_PM)
    parser.add_argument(
        "--amplitude-ratio", type=float, default=DEFAULT_AMPLITUDE_RATIO,
        help=f"|B_θ(r_inner)/B_θ(r_outer)|. Default: {DEFAULT_AMPLITUDE_RATIO}.",
    )
    parser.add_argument("--r-inner", type=float, default=DEFAULT_R_INNER)
    parser.add_argument("--r-outer", type=float, default=DEFAULT_R_OUTER)
    parser.add_argument("--N-r", type=int, default=DEFAULT_N_R)
    parser.add_argument("--N-theta", type=int, default=DEFAULT_N_T)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--order-r", type=int, default=DEFAULT_ORDER_R)
    parser.add_argument("--order-theta", type=int, default=DEFAULT_ORDER_THETA)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--threads-per-worker", type=int, default=DEFAULT_THREADS_PER_WORKER,
    )
    parser.add_argument(
        "--heartbeat", type=float, default=DEFAULT_HEARTBEAT_SECONDS,
    )
    parser.add_argument(
        "--first-light", action="store_true",
        help="Single cheap cell (m=3, Le=0, N=24×24).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
    )
    args = parser.parse_args()

    if args.first_light:
        m_values = [3]
        # Include both Le=0 (hydro recovery) and Le > 0 (magnetic
        # coupling) so first-light actually exercises the new code path.
        Le_values = [0.0, 0.05]
        args.N_r = 24
        args.N_theta = 24
        args.Ek = 3e-3   # well-resolved at N=24
    else:
        m_values = list(args.m_values)
        Le_values = list(args.Le_values)

    cells_list = [(m, Le) for m in m_values for Le in Le_values]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "sweep.log"
    log, log_handlers = _setup_logger(log_path)

    mp_ctx = mp.get_context("spawn")
    from logging.handlers import QueueListener
    log_queue = mp_ctx.Queue(-1)
    log_listener = QueueListener(log_queue, *log_handlers, respect_handler_level=True)
    log_listener.start()

    log.info("=" * 78)
    log.info("Solar eigenproblem + B₀ : Le sweep")
    log.info("=" * 78)
    log.info(f"output_dir       : {args.output_dir}")
    log.info(f"jobs             : {args.jobs}")
    log.info(f"threads/worker   : {args.threads_per_worker} "
             f"(total ≤ {args.jobs * args.threads_per_worker} cores)")
    log.info(f"Ek               : {args.Ek:.3e}")
    log.info(f"Pm               : {args.Pm}")
    log.info(f"amplitude_ratio  : {args.amplitude_ratio}  (|B| at r_inner / at r_outer)")
    log.info(f"r_inner, r_outer : {args.r_inner}, {args.r_outer}")
    log.info(f"N_r × N_θ        : {args.N_r}×{args.N_theta} (DOF={8*args.N_r*args.N_theta:,})")
    log.info(f"order_r, order_θ : {args.order_r}, {args.order_theta}")
    log.info(f"k (per solve)    : {args.k}")
    log.info(f"m values         : {m_values}")
    log.info(f"Le values        : {Le_values}")
    log.info(f"total solves     : {len(cells_list)}")

    if args.dry_run:
        log.info("--dry-run: exiting without compute.")
        log_listener.stop()
        return 0

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
    n_done = 0; n_skipped = 0; n_failed = 0

    # Determine pending solves (skip those already on disk).
    pending = []
    for (m, Le) in cells_list:
        out_path = _cell_output_path(args.output_dir, m, Le)
        if _cell_done_and_valid(out_path):
            n_skipped += 1
            log.info(f"  skip m={m}, Le={Le:.0e}: {out_path.name} exists")
            continue
        pending.append((m, Le, out_path))

    if not pending:
        log.info("Nothing to do.")
        log_listener.stop()
        return 0

    log.info(f"Dispatching {len(pending)} solves "
             f"(jobs={args.jobs}, threads/worker={args.threads_per_worker})...")
    with ProcessPoolExecutor(
        max_workers=args.jobs, mp_context=mp_ctx,
        initializer=_worker_init,
        initargs=(log_queue, args.threads_per_worker),
    ) as ex:
        fut2key = {}
        for (m, Le, out_path) in pending:
            fut = ex.submit(
                _solve_one_cell_safe,
                m, Le, args.N_r, args.N_theta,
                args.Ek, args.Pm, args.amplitude_ratio,
                args.r_inner, args.r_outer,
                args.k, args.order_r, args.order_theta,
                str(args.data_dir), str(out_path),
                args.heartbeat,
            )
            fut2key[fut] = (m, Le)

        for fut in as_completed(fut2key):
            m, Le = fut2key[fut]
            r = fut.result()
            if r.get("ok"):
                n_done += 1
                log.info(
                    f"  m={m:>2} Le={Le:.0e} OK  "
                    f"compile={r['compile_time']:5.1f}s "
                    f"solve={r['solve_time']:6.1f}s  "
                    f"rss={r['rss_peak_kb']/1024/1024:5.2f}GB  "
                    f"Rossby: ω={r['rossby_omega']:+.5f}, γ={r['rossby_gamma']:+.5f}"
                )
                manifest[f"m{m:02d}_Le{Le:.4e}"] = {
                    "m": m, "Le": Le,
                    "N_r": args.N_r, "N_theta": args.N_theta,
                    "Ek": args.Ek, "Pm": args.Pm,
                    "compile_time": r["compile_time"],
                    "solve_time": r["solve_time"],
                    "rss_peak_kb": r["rss_peak_kb"],
                    "output": Path(r["output_path"]).name,
                    "rossby_omega": r["rossby_omega"],
                    "rossby_gamma": r["rossby_gamma"],
                    "t_lu_decompose": r["t_lu_decompose"],
                    "t_arpack": r["t_arpack"],
                    "n_arpack_matvec": r["n_arpack_matvec"],
                }
            else:
                n_failed += 1
                log.error(
                    f"  m={m:>2} Le={Le:.0e} FAILED: "
                    f"{r['error_type']}: {r['error_msg']}"
                )
                manifest[f"m{m:02d}_Le{Le:.4e}"] = {
                    "m": m, "Le": Le,
                    "error": f"{r['error_type']}: {r['error_msg']}",
                }
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

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
