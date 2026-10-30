# spdee — Spherical Partial Differential Equation Eigensolver

Last updated: 2026-10-30

A modular eigensolver for linear PDEs in spherical geometry, separable in
the azimuthal coordinate `φ` so that each Fourier mode `m` yields an
independent generalized eigenvalue problem `L x = λ R x`.

The design separates the **symbolic problem statement** (backend-agnostic)
from the **discretization** (pluggable backends), with a shared
generalized eigenvalue solver layer. Users write equations once and
pick a discretization at solve time.

## Status (2026-09-09)

Four numerical cores and a shared solver layer, all reachable from
equations entered as strings (`spdee.Problem`):

| core | module | state |
|---|---|---|
| Dedalus spectral (shell / ball) | `spdee.backends.dedalus` | IR → Dedalus v3 EVP → `(L_min, M_min)`; shell operators identical to the solar-eigenproblem cross-validation Dedalus code to roundoff; ball: θ-dependent coefficients assembled by column probing (Dedalus has no meridional ball NCC yet, see `docs/dedalus_ball_meridional_ncc.md`), Bessel and Zhang benchmarks to 1e-11 (`tests/test_dedalus_ball.py`) |
| spin-weighted harmonics × radial FD | `spdee.backends.sh_fd` | analytic angular operators (Vasil et al. 2019), Fornberg radial stencils; certified Rossby mode within 3e-4; ball via mirrored ghost nodes with the spin-swap regularity rule (`docs/ball_sh_fd.md`), benchmarks at the FD order |
| staggered Arakawa-C (scheme of Bekki et al. 2022) | `spdee.backends.fd2` (`discretization="staggered"`) | one task-driven assembler (`staggered_general.py`, `docs/staggered_general_design.md`): the walker's tasks are classified (evolution, saddle point, involution, tensor definition) and dispatched to mimetic blocks, and a system it cannot pair is refused rather than guessed. Hydro, Boussinesq, anelastic, fully compressible and constrained-transport MHD on the shell (div b = 0 exact), every block certified entrywise against an independent implementation; ball hydro with the origin-ring closure (`docs/ball_staggered.md`), second-order benchmarks; CT MHD on the ball not implemented |
| collocated FD (orders 2/4/6) | `spdee.backends.fd2` (default) | full operator zoo; origin rows from symbolic L'Hospital limits (`docs/ball_collocated.md`), Bessel, Poisson and Zhang benchmarks at the FD order; carries the usual collocated checkerboard modes |

Solvers (`spdee.solvers`): ARPACK or bbjd shift-invert with SuperLU,
FPLUS (int64, selectable pivoting), cuDSS or CuPy factorizations; dense
full-spectrum census on CPU or GPU (cuSOLVER Xgeev). FPLUS and bbjd (JDQZ
with the harmonic test space) are the separate FPLUS and BBJD libraries,
built on their own and named by `SPDEE_FPLUS_PATH` / `SPDEE_BBJD_PATH`
(`spdee.solvers.external`, `docs/requirements.md`).

Tests, from the repository root with `PYTHONPATH=.`:
`python -m pytest tests/ -q --ignore=tests/test_evp_zhang_2001.py` is the
fast suite (~15 min with a discrete GPU); `-m "slow or not slow"` adds the
624 operator-coverage cells and the Zhang staggered classes.  The Dedalus and
GPU tests skip when the dependency is absent, and
`SPDEE_NO_GPU=1` runs the whole tree on the CPU; the FPLUS and bbjd tests
run when `SPDEE_FPLUS_PATH` / `SPDEE_BBJD_PATH` name builds.  `tests/test_analytic_evp.py`
holds the analytic eigenvalue benchmarks of all four cores
(`docs/analytic_benchmarks.md`); `scripts/operator_coverage.py`
regenerates `docs/OPERATOR_COVERAGE.md`.

Sign conventions were audited against the solar-eigenproblem cross-validation codes on
2026-09-09: the buoyancy term in the Figure-2 / Step-4 builders and the
Lorentz force in the MHD builders had reversed signs and are corrected;
results produced with the collocated or Dedalus paths before that date
carry the old signs.

### Headline capabilities (added since the previous README revision)

* **Aux-free composed MHD operators**: `Cross(Curl(unknown), bg)`
  (Lorentz force) and `Curl(Cross(unknown, bg))` (induction equation)
  are dispatched as direct matrix products of the elementary Curl and
  Cross blocks. No auxiliary unknowns at production scale.
* **Solar eigenproblem + magnetic field application** (`spdee.applications.solar_eigenmodes`):
  divergence-free dipolar B₀ on the shell domain of the solar eigenproblem with the
  30 kG → 1 G amplitude profile, momentum-equation Lorentz force,
  induction equation, and perfect-conductor BCs.
* **Le sweep driver** (`scripts/sweep_solar_mhd_Le.py`): production
  driver for studying how solar inertial Rossby modes shift with
  background magnetic-field amplitude. Quadratic Le² scaling
  confirmed empirically across 5 orders of magnitude in Le.

### Capability summary

* **IR**: full symbolic expression layer with scalar, vector, and
  rank-2 tensor fields; user-facing physical components, internal
  V19 spin basis; products and contractions of background × unknown
  combinations enforced linear at construction.
* **Backend**: `fd2`, fourth-order standard centered FD in r and θ,
  with parity-aware origin closure for ball geometry and standard
  one-sided closure for shell geometry. Sparse-banded operator
  builders throughout, including for composed operators like
  `Grad(Div(v))` and `Dot(bg_vec, tensor_expression(v))` so no
  auxiliary state is introduced.
* **EVP**: shift-invert ARPACK with a banded LU front-end. Optional
  spectrum filtering via `evp_filter.py`. Gauge elimination for
  pressure-like Lagrange multipliers.
* **Backgrounds**: a hierarchy of `BackgroundComponent` subtypes
  (`SymPyComponent`, `NumericComponent`, `CompositeComponent`)
  unifying analytic and tabular sources. Includes a Model-S solar
  adapter under `spdee/backgrounds/solar.py` (solar-eigenproblem replication).

### Operator coverage

For 4th-order standard FD in all four `(stencil × geometry)` contexts
considered during early development. The current code path is "standard
FD" across both shell and ball geometries.

| Operator                                                        | r-power | Output | Status     |
|-----------------------------------------------------------------|---------|--------|------------|
| `Lap(scalar)`, `LapH(scalar)`                                   | 2       | scalar | ✓          |
| `Lap(vector)`                                                   | 2       | vector | ✓          |
| `Grad(scalar)`                                                  | 1       | vector | ✓          |
| `Div(vector) → scalar`                                          | 1       | scalar | ✓          |
| `Curl(vector)`                                                  | 1       | vector | ✓          |
| `Div(tensor_expr) → vector`                                     | 2       | vector | ✓          |
| `Grad(Div(unk_vec))`                                            | 2       | vector | ✓          |
| `Cross(bg_vec, unk_vec)`                                        | 0       | vector | ✓          |
| `Dot(bg_vec, unk_vec) → scalar`                                 | 0       | scalar | ✓          |
| `Dot(bg_tensor, unk_vec)`                                       | 0       | vector | ✓          |
| `Dot(bg_vec, Grad(unk_vec))` (advection)                        | 1       | vector | ✓          |
| `Dot(bg_vec, tensor_expr(unk_vec))`                             | 1       | vector | ✓          |
| `bg_vec * unk_scalar` (buoyancy)                                | 0       | vector | ✓          |
| `bg_vec * Div(unk_vec)` (trace piece)                           | 1       | vector | ✓          |
| `Identity`, `Source`                                            | 0       | both   | ✓          |
| Multi-operator equations with mixed r-powers (mass conversion)  |         |        | ✓          |

Each composed operator above (`Grad(Div(v))`, `Dot(bg_vec, T(v))`,
`bg_vec * Div(v)`) is dispatched via a dedicated walker pattern that
emits a single sparse block per equation row — **no auxiliary
unknowns** are introduced for these compositions, keeping the state
vector minimal at production resolutions.

See `OPERATOR_COMPOSITION.md` for details on the common-r-power
mass machinery used for multi-operator equations, and
`docs/OPERATOR_COVERAGE.md` for the generated operator coverage
matrix: every IR operator against a manufactured solution on every
(core, geometry), with the measured order or the core's refusal.

### Comparison utilities

For cross-code comparison (e.g., spdee vs. Bekki et al. 2022), spdee
exposes:

* **Richardson extrapolation** of a target eigenvalue to the
  continuum limit. The same pattern as
  ``TestCompressibleStep4::test_richardson_extrapolation_to_continuum``
  takes a small ladder of N_r values, fits
  ``ω(N_r) = ω_∞ + C·N_r^{-p}``, and reports the continuum value
  ``ω_∞``. This is the layout-independent quantity that should
  match between any two consistent discretizations.
* **Staggered-grid interpolators** in
  ``spdee/backends/fd2/staggered_io.py``: arbitrary-order Lagrange
  interpolation between spdee's colocated grid and Arakawa-C-style
  staggered (r-face / θ-face / cell-center) grids. Used for
  mode-shape comparison panels and (future) snapshot
  initialization. The solver pipeline is untouched — staggered I/O
  is a comparison utility only.

## Installation and requirements

The minimum tier is numpy, scipy and sympy on Python >= 3.10; it gives the
collocated, staggered and sh_fd cores with ARPACK, SuperLU, the dense host
LU and the dense host census.  Everything else is an optional extra
declared in `pyproject.toml` and described in `docs/requirements.md`:

| extra | adds |
|---|---|
| `dedalus` | the spectral core; Dedalus v3 is built from source by `scripts/install_dedalus_env.sh` (MPI, FFTW, HDF5) |
| `gpu-cuda` / `gpu-cuda12` | CuPy and nvmath-python on an NVIDIA GPU: `factor='cudss'`, `dense(backend='cupy')` with cusolverDnXgeev |
| `gpu-rocm` | CuPy's ROCm build on an AMD GPU: device dense LU (taken by `auto` on a discrete part; measured on an MI300X, see `docs/requirements.md`), geev on the host, no cuDSS; integrated parts (Radeon 890M) need CuPy built from source and are a correctness path where `auto` stays on the host |
| `viz`, `mpi`, `test`, `all` | matplotlib, mpi4py, pytest, everything but ROCm |

```
pip install -e .                       # minimum tier
pip install -e ".[gpu-cuda,ext,viz]"   # a machine with an NVIDIA GPU
python -m spdee.check                  # what this installation can run; exit 0 = minimum tier works
```

Every GPU path degrades to the host with one warning naming the missing
capability (`spdee.solvers.gpu.GPUFallbackWarning`) rather than a
traceback; `factor='auto'` and `backend='auto'` decide by device class,
taking a discrete GPU of either vendor for the dense LU (and cuDSS on
NVIDIA) and staying on the host on an integrated part, where the host
measured faster; `SPDEE_NO_GPU=1` hides the device.  `spdee.solvers.available()`
reports the components present; `python -m spdee.check` prints the full
capability table.

### Environment notes

- numpy on OpenBLAS, not MKL.  Importing pypardiso loads `libmkl_rt`,
  whose FFTW-interface symbols shadow the FFTW Dedalus links, and every
  Dedalus transform then fails with "FFTW could not create plans"; SPDEE
  imports pypardiso only with `SPDEE_USE_PARDISO=1`.  An MKL-based numpy
  also breaks C extensions that call the Fortran `zdotc_` (BBJD goes
  through `cblas_zdotc_sub` for that reason).
- cuDSS returns wrong solves with a success flag on pencils with zero
  diagonal entries unless matching is enabled; the solver layer enables it
  and verifies every GPU factorization on a random right-hand side.
- Keep `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` at 4 or less on a
  shared machine and at 1 in a Dedalus process.

## Quick start

```python
import sympy as sp
from spdee import (
    Background, Unknown, r, theta,
    Grad, Div, Curl, Cross, Dot, Lap, Dt,
    Equation, BoundaryCondition, EquationSystem, Boundary,
    r_hat, theta_hat, phi_hat,
)
from spdee.backends.fd2.grid import make_shell_grid
from spdee.backends.fd2.compiled_problem import compile_problem

# Background fields (depend on r, theta; no phi)
rho0 = Background.Scalar("rho0", expr=sp.exp(-r))
B0 = Background.Vector("B0", v_phi=r * sp.sin(theta))

# Eigenproblem unknowns
v = Unknown.Vector("v")
p = Unknown.Scalar("p")

# Symbolic linear equations
momentum = Equation(
    rho0 * Dt(v) + Grad(p) - Cross(B0, Curl(v)),
    name="momentum",
)
div_free = Equation(Div(v), name="div_free")

# Boundary conditions
no_penetration = [
    BoundaryCondition(Boundary.R_INNER, Dot(r_hat, v), name="vri"),
    BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v), name="vro"),
]

# Assemble the system. Pressure is treated as a Lagrange multiplier
# (gauge unknown) so the EVP doesn't pick up the trivial gauge mode.
system = EquationSystem([momentum, div_free], no_penetration)

# Discretize and solve
grid = make_shell_grid(N_r=32, N_theta=64, r_inner=0.5, r_outer=1.0)
cp = compile_problem(system, grid, m=2, order_r=4, order_theta=4)
result = cp.solve_evp(k=10, sigma=0.0)
print(result.eigenvalues)
```

## Architecture

```
spdee/
  ir/                          # backend-agnostic intermediate representation
    fields.py                  # ScalarField / VectorField / TensorField
    expressions.py             # Expression base, FieldRef, Sum, ScalarMultiple, etc.
    operators.py               # Grad, Div, Curl, Cross, Dot, Lap, LapH, Dt, Transpose
    equations.py               # Equation, BoundaryCondition, EquationSystem
    backgrounds.py             # BackgroundComponent hierarchy + composition
    augment.py                 # Aux unknowns + dropped-bg-component optimizations
    tasks.py                   # OperatorTask vocabulary (19 task kinds)
    tensor_reduce.py           # Symbolic reduction of tensor expressions to
                               #   covariant spin components in v-placeholders
    walker.py                  # Expression-tree walker → OperatorTask sequence
  backends/
    common/                    # cross-backend helpers (e.g. spin-component eval)
    fd2/                       # 4th-order standard centered FD in r and θ
      compiler.py              # Task → sparse matrix assembly
      compiled_problem.py      # User-facing object with solve_evp(...)
      grid.py                  # Shell / ball grids
      stencils.py              # FD stencils (uniform, parity-aware)
      operators.py             # DerivativeBundle wrapping bundle of stencils
      term_classify.py         # Generic spin-term classification kernel
                               #   (decompose → coef-diag → FD action)
      spin_block_assembly.py   # Shared 3×3 spin-block assembly engine
      *_sparse.py              # Sparse builders per operator family
                               #   (all route through spin_block_assembly)
      state_layout.py          # State layout (σ-major × j-major × i_r)
      layout.py                # 2D ↔ flat conversion (Fortran order)
      evp_filter.py            # Spectrum filtering helpers
      staggered_general.py     # Arakawa-C task dispatcher (the assembler)
      staggered_compile.py     # Arakawa-C grids, layout and mimetic blocks
      staggered_mhd.py         # constrained-transport blocks (CTGrid)
    sh_fd/                     # spin-weighted SH x radial FD (angular.py, compiler.py)
    dedalus/                   # IR -> Dedalus v3 EVP (compiler.py)
  solvers/                     # shift_invert / dense; external.py (FPLUS, BBJD), gpu_dense.py, gpu_si.py
  problem.py                   # string entry of equations (Problem)
  backgrounds/
    solar.py                   # Model-S adapter for solar background problems
tests/                         # Production test suite (pytest)
scripts/                       # Runnable drivers: spectrum scans, parameter
                               #   sweeps, plotting (sweep_solar_*, polytrope_spectrum*,
                               #   inviscid_order_sweep*, analyze_sweep, ...)
  SWEEP_README.md              # How to run the inviscid order/resolution sweep
tools/                         # Standalone diagnostic harnesses
data/                          # Solar Model-S data (for solar tests)
OPERATOR_COMPOSITION.md        # Unified compile path / operator composition
docs/                          # Design documents and derivations
  OPERATOR_COVERAGE.md         # Generated operator coverage matrix
                               #   (scripts/operator_coverage.py)
  derivations/                 # Mathematical derivation notes
```

### Where the mathematics is written down

- `docs/staggered_general_design.md`: the C-grid as a discrete de Rham
  complex, placement by grade, Hodge pairing, constraint classification,
  the stage records of the general assembler.
- `docs/ball_collocated.md`, `docs/ball_staggered.md`, `docs/ball_sh_fd.md`:
  the origin and pole treatment of each core, with the regularity rules
  per spin and the closures as implemented.
- `docs/ir_boundary_and_constraints.md`: the composite boundary-row
  record, the constraint classifier, the divergence-cleaning prescription
  and the rule for fixing the multiplier's constant.
- `docs/analytic_benchmarks.md`: every closed form used in the analytic
  eigenvalue benchmarks, with its derivation.
- `docs/dedalus_ball_meridional_ncc.md`: theta-dependent coefficients on
  the Dedalus ball (banded, not dense; blocked by one missing method in
  Dedalus).

### IR conventions

- **Field kinds**: scalar (rank 0), vector (rank 1), rank-2 tensor.
- **Field roles**: `Background` (known, with possible `r`,`θ` dependence)
  vs `Unknown` (eigenproblem variable).
- **Vector/tensor components** are user-facing in **physical**
  `(r̂, θ̂, φ̂)` coordinates. Internally the spdee `fd2` backend
  uses the V19 / Lecoanet spin basis:
  - `v^- = (v_θ − i v_φ) / √2`
  - `v^0  = v_r`
  - `v^+ = (v_θ + i v_φ) / √2`
- **`φ`-dependence is forbidden** in background expressions (would break
  m-separability).
- **Time** appears via the `Dt(...)` marker; eigenmodes separate as
  `X(t) = X̃ e^{λ t}` so `Dt(X) ↦ λ X̃`. The eigenvalue convention is
  `λ = −i ω`, so a retrograde frequency `ω < 0` corresponds to
  `Im(λ) > 0`.
- **Linearity is tracked**: products of two unknowns are rejected at
  construction time.

### Backend conventions

Each backend module exposes a `compile_problem(system, grid, **kwargs)`
function that returns a `CompiledProblem` object with a uniform
`.solve_evp(k=..., sigma=...)` method. Backends register themselves with
`spdee.backends.register_backend` at import time. To add a new backend,
create a subpackage under `spdee/backends/` with its own `compile`
function.

## Notes on the project history

This package replaces an earlier project (`sphereig`) that mixed two
discretization pipelines and tangled the IR with backend-specific
machinery. Documented lessons:

- Compact-FD radial operators do not satisfy the exact identities that
  V19's regularity-class formulas require. spdee initially supported
  both "compact" and "standard" FD families but eventually consolidated
  on standard centered FD of selectable even order.
- Convention conflicts between V19 sections produced subtle sign bugs
  that hid in `m=0` validations but broke general operator identities.
  spdee uses the V19 spin basis throughout the `fd2` backend; the IR
  itself stays in physical components.
- Premature commitment to a single representation throughout made
  backend swap impossible in the prior project. spdee separates these
  concerns cleanly: the IR doesn't know about spin or regularity at
  all; each backend chooses its own internal representation.

## Known limits

Documented behaviour of the discretizations, not defects:

- Staggered: the scalar Laplacian's wall closure is second order whatever
  the interior order, because the divergence is the transpose of the
  gradient and the transpose of a one-sided wall row is not a divergence;
  a summation-by-parts pair would be needed.  The l = 1 toroidal viscous
  ball rates trade cubic exactness at the origin for the constant.
- Collocated: the inviscid inertial band carries damping of order 1e-3 to
  1e-2 on under-resolved modes (Grad and Div are not discrete adjoints);
  three m = 2 Zhang targets collide into complex pairs at order-4 census
  resolutions and are clean at order 6; on a ball at m = 0 the
  divergence-cleaning multiplier needs the mean-zero row as a checkerboard
  pin and carries one spurious mode far from the physical band.
- The staggered inviscid band is dense enough that recovering individual
  inertial-mode targets is not evidence there; the tests assert the error
  and its order instead.

## License

GPL-3.0-or-later (see `LICENSE`).  The FPLUS and BBJD solver libraries
carry the same license and are developed in their own repositories.
