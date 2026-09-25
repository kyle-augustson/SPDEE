# Inviscid order/resolution sweep — run on your machine

`inviscid_order_sweep.py` answers: is the staggered inviscid inertial operator
CLEAN (undamped + convergent) at all interior orders with matched-order boundary
closure, across m and resolution?

## Run
From the spdee repo root (so `spdee` imports), or with PYTHONPATH set to it:

```bash
# full sweep across all cores, all dense (best for the undamping/Re-λ question):
python scripts/inviscid_order_sweep.py --out sweep.csv

# the careful sweep you asked for (m 0..5; orders 2,4,6,8; several N_r, N_θ):
python scripts/inviscid_order_sweep.py \
    --ms 0 1 2 3 4 5 --orders 2 4 6 8 \
    --nr 16 24 32 48 64 --nth 16 24 32 48 64 \
    --jobs <ncores> --out sweep.csv
```

Each (m, order, N_r, N_theta) is one independent task → embarrassingly parallel
over `--jobs` processes. Results stream to CSV as they finish (partial results
survive an interrupt). Independent N_r and N_theta let you see r- vs θ-order
separately; add `--square` to restrict to N_r==N_theta and cut the task count.

DENSE solve is the default and is recommended for the Re(λ) undamping verdict —
the sparse shift-invert path (`--sparse-above K`, sparse when max(N_r,N_θ)>K)
is faster for large N but can mix nearly-degenerate cluster modes and fake a
small Re(λ). The script self-validates sparse vs dense at one small case on
startup and warns if they disagree.

Full-domain dense at N=64 is the expensive end (a few minutes per task,
single-thread); budget accordingly or cap N and use `--sparse-above 48` for the
largest sizes.

## Read
On completion (or any time) summarize:
```bash
python scripts/inviscid_order_sweep.py --summarize sweep.csv
```
Per (m, order) it reports:
- **imag-err rate**: convergence rate of the physical frequency toward the
  analytic toroidal value m/(ℓ(ℓ+1)). ~interior order ⇒ clean convergence.
- **max|Re| persistence**: the undamping verdict. The KEY test is behavior
  UNDER REFINEMENT, not at fixed N:
    - `clean (<1e-9)`     → undamped to machine precision. CLEAN.
    - `scatter ~1e-3`     → small Re appears/vanishes with N → mode-selection
                            ARTIFACT (what the quick tests showed), not a defect.
    - `PERSISTS ~…`       → Re fails to shrink / grows with N → GENUINE spurious
                            growth: the boundary closure breaks the skew/energy
                            structure at that order. *** This is the thing we're
                            hunting. ***

m=0 is degenerate (analytic freq 0): only the Re-survey is meaningful there;
imag-err is left NaN by design.

## What the columns mean (CSV)
m, order, N_r, N_theta, method(dense/sparse),
im_lowest, re_lowest (targeted mode),
imag_err_rel (|Im-analytic|/analytic),
band_max_abs_re (max|Re λ| over |Im|<1 modes — the undamping survey),
n_spurious_pos (# modes with Re>1e-6),
seconds, error.

## Context (from the session that produced this)
Quick in-sandbox tests at m=2 showed: 2nd-order undamped to ~1e-16 (matches the
verified handoff property); higher orders showed scattered ~1e-3 Re(λ) at
ISOLATED N that vanished at neighboring N → consistent with mode-selection
artifacts, NOT persistent spurious growth. This sweep is to confirm that across
m=0..5 and a real N-ladder before concluding "inviscid is clean at all orders."
If it confirms, the high-order DIFFICULTY is specific to the VISCOUS
curl-curl-composition wall closure (the transposed one-sided stencil / missing
surface term), not a general staggered-boundary obstruction.

---

## MPI version (`inviscid_order_sweep_mpi.py`)

Same science (it imports the solve/summary from `inviscid_order_sweep.py` — one
source of truth). Parallelism model is what you asked for: **MPI over the
independent tasks, 2 BLAS threads per rank.**

```bash
# 32 ranks x 2 BLAS threads == 64 cores, full sweep, all dense:
OMP_NUM_THREADS=2 mpirun -n 32 python scripts/inviscid_order_sweep_mpi.py \
    --ms 0 1 2 3 4 5 --orders 2 4 6 8 \
    --nr 16 24 32 48 64 --nth 16 24 32 48 64 --out sweep.csv
```

Launch with `n_ranks ~ ncores / 2`. Each rank streams to `sweep_rank{r}.csv`
(no parallel-write contention); after a barrier, rank 0 merges them into
`sweep.csv` and prints the summary. Per-rank files are kept.

- BLAS threads are pinned at the TOP of the file (before numpy import, since
  OpenBLAS/MKL read the env vars at import). Default 2; setting
  `OMP_NUM_THREADS` in the shell overrides. Set it to match `-n` so ranks don't
  oversubscribe cores (`n_ranks * threads == ncores`).
- `spdee` does not use MPI internally (checked), so mpi4py owns the communicator
  with no conflict; each rank is an independent serial dense solve.
- Round-robin task assignment (`tasks[rank::size]`) balances load — cost grows
  with N, and consecutive tasks share an N, so striding spreads the big ones
  across ranks.
- Summarize-only needs no MPI: `python scripts/inviscid_order_sweep_mpi.py --summarize sweep.csv`
- Requires `mpi4py` and an MPI launcher (mpich/openmpi). The serial
  `inviscid_order_sweep.py` (ProcessPoolExecutor, single-node shared-memory)
  remains as a no-MPI fallback.
