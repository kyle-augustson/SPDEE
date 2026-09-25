#!/usr/bin/env python3
"""
Inviscid staggered inertial-EVP order/resolution sweep -- MPI VERSION.

Same science as inviscid_order_sweep.py (see its docstring for the full
explanation of metrics, the dense-vs-sparse undamping caveat, and how to read
the results). This version distributes the independent (m, order, N_r, N_theta)
tasks over MPI ranks, with each rank running its dense LAPACK solve on a small,
fixed number of BLAS threads (default 2).

PARALLELISM MODEL
-----------------
  * MPI over TASKS: rank r handles tasks r, r+size, r+2*size, ...  (round-robin
    so the per-rank load is balanced even though task cost grows with N).
  * THREADS within a rank: each rank's dense eig uses OMP_NUM_THREADS (default
    2) BLAS threads. So total core usage ~ size * threads. Launch with
        n_ranks ~ ncores / threads
    e.g. on a 64-core node with 2 threads/rank: mpirun -n 32.

  IMPORTANT: BLAS thread count is fixed at the TOP of this file, BEFORE numpy is
  imported, because OpenBLAS/MKL read these env vars at import time. Setting
  OMP_NUM_THREADS in the shell also works and overrides the default here.

OUTPUT
------
  Each rank streams its results to  <out_stem>_rank{r}.csv  (no parallel-write
  contention). After a barrier, rank 0 concatenates them into <out_stem>.csv and
  prints the summary. Per-rank files are kept (delete if you don't want them).

USAGE
-----
  # 32 ranks, 2 BLAS threads each (== 64 cores), full sweep, all dense:
  OMP_NUM_THREADS=2 mpirun -n 32 python inviscid_order_sweep_mpi.py \
      --ms 0 1 2 3 4 5 --orders 2 4 6 8 \
      --nr 16 24 32 48 64 --nth 16 24 32 48 64 --out sweep.csv

  # square grids only, sparse above N=48 for the large tail:
  OMP_NUM_THREADS=2 mpirun -n 16 python inviscid_order_sweep_mpi.py \
      --square --sparse-above 48 --out sweep.csv

  # summarize an existing merged CSV (no MPI needed; run with -n 1 or plain python):
  python inviscid_order_sweep_mpi.py --summarize sweep.csv

REQUIRES: mpi4py, numpy, scipy, and spdee importable (run from repo root or set
PYTHONPATH). spdee does NOT use MPI internally (checked), so mpi4py owns the
communicator with no conflict.
"""
from __future__ import annotations

# --- BLAS thread pinning MUST happen before numpy/scipy import -------------
import os
_DEFAULT_THREADS = "2"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, _DEFAULT_THREADS)
# ---------------------------------------------------------------------------

import argparse
import csv
import glob
import time

# Path setup — script lives in scripts/, code lives in spdee/. Also keeps
# the sibling `inviscid_order_sweep` import working regardless of cwd.
import sys
from pathlib import Path as _Path
_THIS_DIR = _Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Reuse the science from the serial module so there's ONE source of truth for
# the solve, the analytic target, and the summary. (Same directory import.)
try:
    from inviscid_order_sweep import (
        run_one, summarize, FIELDS, validate_solvers)
except ImportError:
    raise SystemExit(
        "inviscid_order_sweep.py must be in the same directory (it holds the "
        "shared solve/summary logic).")


def build_tasks(args):
    nr_nth = [(nr, nth) for nr in args.nr for nth in args.nth
              if (not args.square or nr == nth)]
    return [(m, o, nr, nth)
            for m in args.ms for o in args.orders for (nr, nth) in nr_nth]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ms", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--orders", type=int, nargs="+", default=[2, 4, 6, 8])
    ap.add_argument("--nr", type=int, nargs="+", default=[16, 24, 32, 48])
    ap.add_argument("--nth", type=int, nargs="+", default=[16, 24, 32, 48])
    ap.add_argument("--square", action="store_true")
    ap.add_argument("--sparse-above", type=int, default=0)
    ap.add_argument("--out", type=str, default="inviscid_sweep.csv")
    ap.add_argument("--summarize", type=str, default=None)
    ap.add_argument("--no-validate", action="store_true")
    args = ap.parse_args(argv)

    # summarize-only path: no MPI required
    if args.summarize:
        summarize(args.summarize)
        return

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        nthreads = os.environ.get("OMP_NUM_THREADS", "?")
        print(f"[mpi] {size} ranks x {nthreads} BLAS threads "
              f"(~{size}*{nthreads} cores)")
        if not args.no_validate:
            validate_solvers()  # sparse-vs-dense sanity, on rank 0 only

    comm.Barrier()

    tasks = build_tasks(args)
    if rank == 0:
        print(f"[mpi] {len(tasks)} tasks, round-robin over {size} ranks "
              f"(method: {'sparse>'+str(args.sparse_above) if args.sparse_above else 'all-dense'})")

    # round-robin assignment: balances load since cost grows with N and
    # consecutive tasks share an N (so striding spreads the big ones out)
    my_tasks = tasks[rank::size]

    stem = args.out[:-4] if args.out.endswith(".csv") else args.out
    my_csv = f"{stem}_rank{rank}.csv"
    t0 = time.perf_counter()
    with open(my_csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        fh.flush()
        for i, (m, o, nr, nth) in enumerate(my_tasks, 1):
            rec = run_one(m, o, nr, nth, args.sparse_above)
            wr.writerow({k: rec.get(k, "") for k in FIELDS})
            fh.flush()
            tag = f"m{m} o{o} {nr}x{nth}"
            if rec["error"]:
                print(f"[rank {rank}] ({i}/{len(my_tasks)}) {tag} ERROR {rec['error']}")
            else:
                print(f"[rank {rank}] ({i}/{len(my_tasks)}) {tag} "
                      f"Im={rec['im_lowest']:.5f} Re={rec['re_lowest']:.1e} "
                      f"bandRe={rec['band_max_abs_re']:.1e} ({rec['seconds']:.1f}s)")

    comm.Barrier()
    elapsed = comm.reduce(time.perf_counter() - t0, op=MPI.MAX, root=0)

    if rank == 0:
        # merge per-rank CSVs into the final one
        merged = args.out
        rows = []
        for f in sorted(glob.glob(f"{stem}_rank*.csv")):
            with open(f) as fh:
                rows.extend(list(csv.DictReader(fh)))
        with open(merged, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, "") for k in FIELDS})
        print(f"[mpi] done in {elapsed:.0f}s (slowest rank) -> {merged} "
              f"({len(rows)} rows; per-rank files kept)")
        summarize(merged)


if __name__ == "__main__":
    main()
