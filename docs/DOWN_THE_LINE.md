# DOWN_THE_LINE markers

Points to revisit when the corresponding follow-on work begins. Each
marker captures an architectural decision that's "good enough for now"
but may need refinement under future requirements.

## Markers

### M1: BC abstraction for tau methods (2026-05-13)

**Where**: BC walker design (`spdee/ir/walker.py`, BC dispatch in
`spdee/backends/<name>/compiler.py`).

**Current state**: BC walker produces `(constraint_matrix R, rhs_vector)`
per BC. fd2 backend installs via row replacement. Tau-method backends
(sh_fd, dedalus) are expected to install the *same* `(R, rhs)` by
appending rows (and optionally columns for tau parameters).

**Risk**: classical tau augments the system with both rows AND columns
(the tau coefficients). The walker's `(R, rhs)` abstraction may not
expose what tau needs to size its column augmentation, or to associate
tau columns with specific BCs. The column-augmentation step might leak
backend-specific concepts into the abstract layer if not designed
carefully.

**When to revisit**: when implementing the sh_fd or dedalus backend's
BC installation. Specifically, when we first try to install a Dirichlet
BC via tau and need to size the column block.

**Possible refinements**:
- Add a per-BC `tau_columns` field describing tau augmentation needed.
- Or: tau backends own their column gymnastics internally without IR
  involvement, treating `(R, rhs)` as the row-level constraint only.

### M2: Strategy C (implicit operator elimination) for performance (2026-05-13)

**Where**: Operator composition handling in the equation walker /
compiler.

**Current state**: Compositions like `Cross(v0, Curl(v))` will use
state augmentation (Strategy A) — introduce auxiliary unknown `w =
Curl(v)` with constraint `A_curl @ w − B_curl @ v = 0`, then point-
multiply by v0 on w.

**Risk**: state augmentation grows the matrix size. For deeply
composed operators or large systems, this becomes a memory and
solver-time concern.

**When to revisit**: when Stage 4 eigenproblem performance becomes a
limiting factor, or when specific problems need denser operator chains
than naive augmentation handles efficiently.

**Possible refinements**:
- **Strategy C** (implicit elimination at LU level): exploit the
  Kronecker-like structure of `A_op = A_θ @ A_r @ diag(...)` so that
  applying `A_op^{-1}` is two banded 1D solves. Use a custom
  `LinearOperator` for the solver instead of forming the matrix.
- **Strategy D** (direct closed-form operators): for hot compositions
  with known closed forms (e.g. `Cross(Omega, v)` for Coriolis, or
  more complex cases), derive and code the direct sparse matrix.

### M4: Multiple differential operators in one equation (2026-05-13)

**Where**: fd2 compiler dispatch (and future backend compilers).

**Current state**: The compiler assumes at most ONE differential
operator per equation. When determining the "equation mass matrix"
`A_eq`, it takes either `I` (no diff operators) or `A_op` of the sole
diff operator. If two diff operators appear (e.g. induction's `Curl`
and `Lap` on the same unknown), the compiler raises with a pointer
to state augmentation.

**When to revisit**: when we start writing induction-like equations
for MHD eigenproblems. Expected to coincide with extending the walker
to handle vector unknowns and intermediate state augmentation.

**Possible refinements**:
- **State augmentation in the walker**: detect multi-operator equations
  and rewrite as multiple equations + constraints (`w = Op2(...)` aux).
  This pushes the complexity into the walker, keeps the compiler simple.
- **Mass-product approach**: directly form `A_op1 @ A_op2` as `A_eq`,
  multiply all terms accordingly. Bandwidth doubles; performance risk.
- The state-augmentation path is preferred (Strategy A); keep this
  option only if augmentation becomes too costly.

**Regression marker**: if we go down the state-augmentation route and
it gets hairy, we can return to this point and reconsider the
mass-product alternative.

### M5: Stencil choice (compact vs standard) at large N (2026-05-13)

**Where**: `spdee/backends/fd2/operators.py` (`DerivativeBundle`),
exposed via `stencil="compact"` or `stencil="standard"` argument to
`compile_problem` / `compile_fd2`.

**Current state**: `stencil="compact"` is the default. Both work.

**Observation from N=512 stress tests**:
- At large N, the compact stencil accumulates round-off from its
  implicit tridiagonal solve, visible as a rate degradation
  (4 → 3 at N=256→512) in absolute error around ~5e-12.
- The standard stencil has comparable absolute accuracy at moderate N
  and avoids the implicit-solve round-off step, but uses a wider
  band in the assembled matrix.
- For BVPs at large N, the standard stencil may be preferred (simpler
  matrices, less LU round-off).
- For eigenproblems, conditioning of `A - σB` may favor compact (because
  compact "spreads" the derivative information into both `A_op` and
  `B_op`, potentially improving conditioning). Unverified.

**When to revisit**: when Stage 4 eigenproblem work begins, run
side-by-side comparisons at multiple N values:
- Memory usage of A_total and B_total.
- Time per LU factorization (or per shift-invert step).
- Condition number of `A_total - σ * B_total` at the shift σ.
- Accuracy of recovered eigenvalues.

Decide per-problem-class which stencil is the better default.

**Possible refinements**:
- Add a small utility `compare_stencils(system, grid, m)` that runs the
  same problem under both stencils and reports a benchmark table.
- Document recommendations in the user-facing docs (when those exist).
- Per-problem default selection (e.g. switch to standard when
  `N_r * N_theta > threshold`).

### M3: Ball-mode geometry — scalar any m ✓, vector m=0 ✓, vector |m|=1 ✓, vector |m|≥2 ✓; tensor remaining (2026-05-14, updated)

**Where**: `spdee/backends/fd2/stencils.py`, `spdee/backends/fd2/operators.py`,
`spdee/backends/fd2/compiler.py`, `spdee/backends/fd2/compiled_problem.py`.

**Current state (2026-05-14, after |m|=1 vector α-DOF anchor patch)**:

1. **Parity-aware compact stencils implemented** in `stencils.py`:
   `compact_d1_uniform_parity(N, h, parity)` and
   `compact_d2_uniform_parity(N, h, parity)`. Both ±1 parity supported,
   tested for 4th-order convergence (see `tests/test_stencils_parity.py`,
   12 tests).

2. **`DerivativeBundle` exposes parity-aware accessors**: `parity_d1_r(parity)`
   and `parity_d2_r(parity)`, cached per parity value. Reject shell grids.

3. **Scalar Laplacian at any m implemented** in
   `spdee/backends/fd2/operators.py::_lap_scalar_ball`. Uses
   the r²-multiplied form with no 1/r factors:

       r² Lap f = r² D2_r f + 2r D1_r f + D2_θ f + cot θ · D1_θ f
                - (m²/sin²θ) f

   Parity = (-1)^|m| on the radial stencils. A_op = diag(r²) with
   i_r=0 rows overwritten as identity placeholders.

4. **Vector Laplacian at m=0 implemented** in `_lap_vector_ball_m0`.
   r²-multiplied form per spin component with the m=0 angular
   couplings; all spins use even parity since the leading origin
   behavior (r⁰ Cartesian value) is r-even.

5. **Vector Laplacian at m≠0 implemented** in `_lap_vector_ball_m_nonzero`.
   Same r²-multiplied pattern as m=0, with the full-m terms:
   `-(m²/sin²θ)` in the per-block scalar piece, m/sin θ shifts in
   the off-diagonal angular operators, and `±2m cos θ/sin²θ` on the
   v^± self-couplings. **Parity = +1 (even) on all three spin
   components** (corrected this session from the original
   `(-1)^|m|` choice, which was wrong for |m|=1 — see point 9).

6. **Compiler accepts**:
   - Ball + scalar: any m. ✓
   - Ball + vector: any m, with correct origin treatment per regime
     (m=0, |m|=1, |m|≥2). No `UserWarning` emitted at any m. ✓
   - Ball + tensor: rejected (NotImplementedError).

7. **Origin regularity installed in `CompiledProblem`**:
   - Scalar m=0: ℓ=0-projected L'Hospital anchor + N_θ−1 equality
     rows (`_apply_ball_scalar_origin_regularity_m0`).
   - Scalar m≠0: Dirichlet-zero on all (i_r=0, j) rows.
   - Vector m=0: ℓ=0-projected L'Hospital on Cartesian z-component
     + 3N_θ−1 angular-structure Dirichlet rows
     (`_apply_ball_vector_origin_regularity_m0`).
   - **Vector |m|=1: ℓ=0-projected L'Hospital on Cartesian v_α
     projection + 3N_θ−1 angular-structure Dirichlet rows
     (`_apply_ball_vector_origin_regularity_m_abs1`, added this
     session). Recovers the in-plane Cartesian-vector α DOF at the
     origin. See point 8.**
   - Vector |m|≥2: Dirichlet-zero on all (σ, i_r=0, j) rows
     (`_apply_ball_origin_regularity_mnonzero`, vector branch).

8. **|m|=1 vector α-DOF anchor (M3-followup, completed this
   session)**:

   Key derivation: v_α := (x̂∓iŷ)/√2 · v_full is an *m=0
   Cartesian-smooth scalar* (the e^{∓iφ} phase carried by the
   Cartesian projector cancels the m=±1 phase of v_full). So the
   standard m=0 L'Hospital `3 v_α,rr(0,θ) = Lap(v_α)(0,θ)` applies
   directly, and the standard ℓ=0 projection extracts a single
   anchor equation that determines α — no new ℓ=1 associated-
   Legendre projection machinery needed.

   Implementation: in `_apply_ball_vector_origin_regularity_m_abs1`,
   1 anchor row computes the ℓ=0-projected L'Hospital combining the
   three spin components via the v_α coefficients
   `(sin θⱼ/√2, (cos θⱼ+1)/2, (cos θⱼ−1)/2)` for m=+1
   (with v⁻/v⁺ role-swapped for m=−1), and 3N_θ−1 angular Dirichlet
   rows tie all other (σ, i_r=0, j) state values to the anchor
   variable v⁻(0, θ_0) for m=+1 (or v⁺(0, θ_0) for m=−1) via the
   spin patterns. The anchor variable is well-conditioned because
   `(cos θ_0 + 1) ≈ 2` on the cell-centered θ grid.

   See:
   - `DERIVATION_M1_ORIGIN.md` — original derivation note
     (initial framing using ℓ=1 weights was a dead end; the
     correct framing — v_α as an m=0 Cartesian-smooth scalar —
     is recorded in the implementation docstring).
   - `scratch/derive_m1_origin_*.py` — spin pattern derivations.
   - `scratch/derive_m1_anchor_corrected.py` — proof that
     Lap_{m=0}(v_α) = project(Lap_{m=+1}(v)) (the projection
     commutes with the Laplacian).
   - `scratch/derive_m_minus_1_anchor.py` — m=−1 spin patterns.

9. **Parity bug fix in `_lap_vector_ball_m_nonzero`**: the original
   code used `parity = (-1)^|m|` on all three spin components — the
   right choice for scalar mode m, but **wrong for vector spin
   components at |m|=1** because the manufactured field
   v = α(1-r²)(x̂±iŷ)/√2 has r-even leading content in all three
   spin components (not r-odd). Empirical residual test
   (`scratch/test_parity_m1.py`) confirmed: parity=+1 gives a clean
   operator (residual ~4.5e-3 at N=16, smooth-discretization-
   limited), parity=−1 gives residual ~1.57 (catastrophic). Fixed
   to parity=+1 for all spin components, matching
   `_lap_vector_ball_m0`. The pre-existing m=2 test continues to
   pass since the m=2 manufactured field is also r-even.

10. **End-to-end tests pass** (330 tests, +2 new |m|=1 α recovery
    tests added this session, 1 warning test rephrased to
    "no-warning at any m"). Convergence at ~3.9-order observed for
    both m=+1 and m=−1 over N ∈ {12, 16, 24, 32, 48}; α recovered
    to ~5+ digits at N=48 from a manufactured α=2.3.

**Remaining work — tensor ball-mode**: per-σ_total parity,
multiple spin DOFs, tensor origin regularity. Rejected by the
compiler currently.


### T0: Tensor expression IR + reduction primitive (Stage A complete, 2026-05-14)

**Where**: `spdee/ir/operators.py`, `spdee/ir/expressions.py`,
`spdee/ir/fields.py`, `spdee/ir/tensor_reduce.py` (new).

**Status: COMPLETE.** Foundation for tensor expressions in BCs and
in equations (M6 follow-up, T-1 and beyond).

**Scope**: tensor *expressions* only, NOT tensor unknowns. Tensor-typed
objects in the IR exist as expressions composed of vector unknowns and
background fields. The reducer collapses tensor expressions to
per-(σ₁, σ₂) scalar formulas in vector spin components, which downstream
operators (`Div`, `Dot`, BC projections) consume.

**Landed pieces**:

1. **`Transpose` operator** in `ir/operators.py`. Tensor-only argument;
   swaps spin-pair indices in the reducer.

2. **`AsSymmetric` expression wrapper** in `ir/expressions.py`, exposed
   via the `.as_symmetric()` method on any tensor `Expression`. Triggers
   runtime symbolic verification at reduction time.

3. **`symmetric` flag on `TensorField`** with construction-time
   verification for SymPy-backed background fields. Exposed via
   `Background.Tensor(..., symmetric=True)`.

4. **`reduce_tensor_expression(expr, m) → ReducedTensor`** in
   `ir/tensor_reduce.py`. The load-bearing primitive. Handles:
   - `Grad(vector_expression)` — leaf case; computes all 9 spin
     components via the V19-basis formula table derived in
     `scratch/derive_grad_v_9components.py`.
   - `Transpose(tensor_expr)` — swaps spin-pair indices.
   - `Sum(tensor_a, tensor_b)` — adds per-pair.
   - `Negation(tensor_expr)` — negates per-pair.
   - `ScalarMultiple(scalar, tensor)` — scales per-pair (only
     background scalar supported for now).
   - `AsSymmetric(tensor_expr)` — triggers symmetry verification
     and collapses to 6 unique components.
   - `FieldRef` of a background `TensorField` — uses the field's
     `spin_components()` method.

5. **Structural symmetry detection** in
   `is_structurally_symmetric(expr)`: recognizes `X + Transpose(X)`,
   `Transpose(sym)`, `Sum(sym, sym)`, `ScalarMultiple(c, sym)`,
   `Negation(sym)`, `FieldRef(TensorField(symmetric=True))`,
   and `AsSymmetric(X)` patterns. Uses cheap structural equality
   (not Python `is`) to handle the common case where users write
   `Grad(v) + Transpose(Grad(v))` with two separately-constructed
   `Grad` nodes.

6. **Tests**: `tests/test_tensor_reduce.py` with 29 tests covering
   all 9 Grad(v) spin components, structural symmetry detection,
   `as_symmetric()` success and failure paths, symmetric-background
   verification, Transpose roundtrip, Sum and ScalarMultiple
   composition, and cross-check against the strain-BC derivation.

**Cross-check**: the (±1, 0) components of `Grad(v) + Transpose(Grad(v))`
match the strain-BC formulas derived independently in
`DERIVATION_STRAIN_BC.md` and `scratch/derive_strain_bc.py`.

**Test count after this stage**: 359 passed (330 prior + 29 new).

**Out of scope for T0 (deferred to later stages)**:
- Tensor unknowns as primary state — would require state layout,
  origin regularity in ball mode, tensor task kinds. Reducer raises
  `NotImplementedError` if asked to reduce a `FieldRef` to an unknown
  TensorField.
- `Identity` tensor primitive (would be useful for `-(2/3) I Div(v)`
  trace terms; can be added as another leaf case when needed).
- Tensor Laplacian (`Lap(T)`), Curl(T), etc.

**Next stages**:
- **T1**: BC walker uses the reducer for stress-free BCs
  (`ImpenetrableStressFree(v)` helper + symbolic-form support).
  See `DERIVATION_STRAIN_BC.md`.
- **T2**: equation walker uses the reducer for `Div(tensor_expr)`
  and `Dot(v_bg, tensor_expr)` in vector equations.


### M6: Tensor-projection BCs (stress-free angular) — COMPLETE (2026-05-14)

**Where**: BC walker (`spdee/ir/walker.py`), BC dispatch in
`spdee/backends/fd2/compiled_problem.py`, helper in `spdee/ir/equations.py`.

**Status: COMPLETE, including robustness pass (2026-05-15).**
Stress-free angular boundary conditions are implemented end-to-end on
top of the T0 tensor expression IR, generalized to **all 9** (outer, inner)
unit-vector pair projections.

**Robustness work landed (2026-05-15)**:

- **m≠0 manufactured tests**: m=+1, m=-1, m=2 manufactured solutions
  verify the m-dependent strain BC terms. All converge at ~4th order.
- **D1_θ caching**: `DerivativeBundle.d1_theta_dense` lazy property
  caches the dense angular-derivative matrix; `CompiledMatrices.bundle`
  exposes it. BC dispatch uses the cached version.
- **Variable-spacing radial stencil**: BC dispatch now uses actual
  `r` values for one-sided radial-derivative stencils (no uniform-h
  assumption).
- **General (outer, inner) unit-vector projection**: walker now handles
  all 9 (outer, inner) ∈ {r̂, θ̂, φ̂}² combinations via a single
  V19 dual-pairing helper `_project_tensor_to_scalar`. Derivation
  and physical-basis cross-check in
  `scratch/derive_inner_unit_projection.py`.
- **Left/right-multiplication consistency**: walker correctly identifies
  the (first_index_unit, second_index_unit) pair regardless of whether
  the user writes `Dot(u, Dot(T, w))` or `Dot(Dot(u, T), w)` — both
  compute `u^α T_{αβ} w^β` (same physical contraction).
- **Index-based host_sigma routing**: host_sigma is determined by the
  (first_index_unit, second_index_unit) pair rather than syntactic
  position. Equivalent forms hit the same row in the matrix.
- **V19 covariant convention pinned down**: the reducer's `T_{σ_1, σ_2}`
  is COVARIANT (a fact that was inadvertently used correctly in some
  places but with the wrong sign convention elsewhere — see below).
  Determined and documented in `scratch/audit_v19_convention.py`,
  `scratch/audit_convention_simple.py`,
  `scratch/audit_convention_v_phi.py`, `scratch/audit_walker_phi.py`.
- **Sign-on-i bug fix** in `_project_tensor_to_scalar` (latent bug
  surfaced by audit): the contraction now correctly applies
  `T_{σ_1, σ_2} w^{σ_2}` (covariant·contravariant), not
  `T^{σ_1, σ_2} w_{σ_2}` with sign-flipped w (which was the previous
  incorrect formula). The bug was latent because (a) all stress-free
  BCs use inner=r̂ which has only real spin coefficients, (b) the
  strain tensor is symmetric so the antisymmetric error contribution
  cancels, and (c) typical manufactured fields used in tests had
  v_φ = 0. The bug surfaces only for non-symmetric tensors AND
  v_φ ≠ 0 AND non-r̂ inner units — a combination that's unusual
  enough to escape earlier validation.
- **Host-row collision detection**: when two BCs in a problem target
  the same boundary row (same unknown, host_sigma, location), the
  backend raises a clear `ValueError` rather than silently overwriting.

**Landed pieces**:

1. **New BC task kind** `BC_SCALAR_LINEAR_FROM_TENSOR` in
   `spdee/ir/tasks.py`. Carries a SymPy scalar expression in vector
   spin-component placeholders plus metadata identifying the host
   row.

2. **Walker support** in `spdee/ir/walker.py::_try_parse_tensor_projection_bc`.
   Recognizes the patterns:
   - `Dot(r_hat, Dot(tensor_expr, r_hat))` → constrains `T^{0, 0}`.
   - `Dot(theta_hat, Dot(tensor_expr, r_hat))` → constrains
     `(T^{+1, 0} + T^{-1, 0})/√2`.
   - `Dot(phi_hat, Dot(tensor_expr, r_hat))` → constrains
     `i(T^{-1, 0} - T^{+1, 0})/√2`.
   In each case the tensor expression is reduced via the Stage A
   reducer, the projection coefficient is applied, and a single
   scalar SymPy formula in placeholders is emitted.

3. **Backend assembly** in
   `spdee/backends/fd2/compiled_problem.py::_apply_scalar_from_tensor`.
   Walks the scalar BC expression term-by-term, identifies each
   term's (coefficient, placeholder, action) where action ∈
   {pointwise, ∂_r, ∂_θ}, and assembles a constraint row by:
   - Pointwise terms contribute to a single column.
   - ∂_r terms expand to a 5-point one-sided 4th-order Lagrange
     stencil at the boundary radial index.
   - ∂_θ terms expand to the row of D1_θ at the boundary angular
     index, distributing weight across the radial slice.

4. **Helper class** `ImpenetrableStressFree(v, boundary)` in
   `spdee/ir/equations.py`. Constructs the three BCs (one
   impenetrability + two angular-stress) by writing the symbolic
   strain expression internally. Tested to give identical numerical
   results to the explicit symbolic form.

5. **End-to-end test**: manufactured solution
   v_r = (1 − r²) cos θ, v_θ = −(1 + r²) sin θ, v_φ = 0 with source
   g_r = 2 cos θ, g_θ = −2 sin θ on a ball with stress-free BC at
   r = 1. Converges at ~4th order; recovers v^0 at the origin to
   machine precision at moderate N.

**Test count after this stage**: 369 passed (359 prior + 10 new).

**Limitations** (out of scope for the first cut):

- Single-unknown tensor BCs only. Multi-unknown tensor expressions
  (e.g. coupling two vector unknowns in one BC) raise NotImplementedError.
- Only first-order derivatives of placeholders. Higher derivatives
  would need stencil extensions.
- Only ∂_r and ∂_θ in the SymPy expression; no mixed ∂_r ∂_θ in a single
  term (which would arise from, e.g., `Grad(Grad(v))` constructions
  in BCs).
- ~~Inner unit vector for tensor projection BCs must be `r_hat`.~~
  **Lifted in 2026-05-15 robustness pass.** All 9 (outer, inner) pairs
  of unit vectors are now supported via the V19 dual-pairing
  projection helper.

These limitations are unlikely to bite for standard stress-free BCs
on viscous flows. If needed later, each is a bounded extension to
the existing dispatcher.


### M7 (T-2): Div(tensor_expr) in vector equations — COMPLETE (2026-05-15)

**Where**: equation walker (`spdee/ir/walker.py::_emit_div`), reducer
primitive (`spdee/ir/tensor_reduce.py::div_tensor_spin_components`),
backend dispatcher (`spdee/backends/fd2/compiler.py::_dispatch_div_tensor_vector`).

**Status: COMPLETE.** Div(tensor_expression) inside a vector equation
is supported end-to-end. The user can write
    `Equation(Div(Grad(v) + Transpose(Grad(v))) - f, name="eq")`
and the system compiles and solves to the correct continuous solution.

**Landed pieces**:

1. **V19 spin-basis Div(T) formula derived and validated**:
   - Physical formula in `scratch/derive_div_tensor_physical.py` with
     three sanity checks (constant ẑ, r² ẑ, r ŷ at m=+1).
   - Spin-basis derivation in `scratch/derive_div_tensor_spin.py`.
     A first attempt got the σ ↔ -σ swap on the ±1 output components
     wrong (because the physical-to-spin transformation needs the
     DUAL basis e^σ = e_{-σ} for both input and output conversion);
     fixed and re-validated.
   - End-to-end validation in `scratch/validate_div_tensor_spin.py`:
     5 cases of Div(Grad v) = Lap v including m=+1 with v_φ ≠ 0 and
     m=+2 generic field. All pass symbolically.

2. **Reducer primitive** `div_tensor_spin_components(T_cov, m) -> {σ_1: SymPy}`:
   - Takes the reducer's covariant T_{σ_1, σ_2} output.
   - Returns the contravariant (Div T)^{σ_1} as a SymPy expression
     in T placeholders.
   - The output has first-order T-derivatives only.

3. **New task kind** `TASK_DIV_TENSOR_VECTOR` in `spdee/ir/tasks.py`.
   Carries:
   - `per_sigma_expr_in_T[σ_1]`: SymPy for (Div T)^{σ_1} in T placeholders.
   - `T_in_v[(σ_a, σ_b)]`: SymPy for T_{σ_a, σ_b}(r, θ) in v placeholders.
   - `T_placeholders`, `v_placeholders`: the symbol mappings.

4. **Walker `_emit_div`** in `spdee/ir/walker.py`:
   - Validates that the operand is a tensor expression with a single
     unknown vector field.
   - Reduces the tensor to its covariant spin components (T_in_v).
   - Builds T placeholder SymPy Functions and applies
     `div_tensor_spin_components` symbolically on them (Div-in-T).
   - Emits the new task with the two-stage payload.

5. **Two-stage backend dispatcher** `_dispatch_div_tensor_vector`:
   - **Stage 1**: for each (σ_a, σ_b), classify T_in_v on v placeholders
     (only first-order v-derivatives present) and build M[(σ_a, σ_b)] of
     shape (N, 3N) mapping v's 3-spin column block to the scalar
     T_{σ_a, σ_b} field on the grid.
   - **Stage 2**: for each output σ_1, classify per_sigma_expr_in_T on
     T placeholders (only first-order T-derivatives present) and compose:
     `block[row σ_1] += coef_diag @ Op(action) @ M[(σ_a, σ_b)]`.
   - This decoupling avoids ever needing a second-order term classifier;
     the second-derivative behavior of Div(Grad(v)) emerges naturally
     from the composition `Op(action_T) @ Op(action_v)`.

6. **i_r=0 NaN mask in `_coef_diag_from_sympy`**: 1/r coefficients
   evaluate to inf/NaN at r=0 in ball mode. The standard-FD stencil at
   i_r=1 then references those NaN entries, polluting the matrix. The
   mask zeros out non-finite entries (specifically at i_r=0 in ball
   mode) before they propagate; origin regularity overwrites the
   i_r=0 rows downstream, so this is safe.

7. **Tests** in `tests/test_div_tensor_walker.py` (8 tests):
   - Walker emits correct task kind for both `Div(Grad(v))` and
     `Div(Grad(v) + Transpose(Grad(v)))`.
   - Walker payload contains first-order derivatives only in both
     T_in_v and per_sigma_expr_in_T.
   - End-to-end manufactured solution `v = (1-r²) ẑ` for
     `Div(Grad(v)) - f = 0` converges at rate ≥ 2 (standard FD).
   - At N=48, Div(Grad(v)) and direct Lap(v) solutions agree to
     within 5e-5 (same continuous solution, different discretizations).
   - End-to-end for `Div(strain) - f = 0` (with strain = Grad(v) +
     Transpose(Grad(v))) converges at rate ≥ 1.9.
   - m=+1 manufactured ŷ-direction solution converges.

**Test count**: 393 passing (385 → +8 new Stage T-2 tests).

**Limitations** (out of scope for this cut):
- Single-unknown Div only.
- Standard FD discretization gives 2nd-order convergence; for higher
  order we'd need compact FD stencils with their A_op mass matrix
  threaded through the composition, which is more invasive.
- The i_r=0 mask gives an O(h) accuracy hit at the origin in ball mode,
  but this is dominated by the existing origin-regularity treatment.

**Next stages**:
- **T-3** (if needed): `Dot(v_bg, tensor_expr)` in vector equations.
  Similar two-stage composition but with the inner Dot's vector pulled
  in as a coefficient.
- **Real-equation dogfood**: viscous flow with Div(strain) in the
  equation, possibly coupled to a pressure unknown. The infrastructure
  is now ready.


### M7 v2 (T-2 rev): r²·Div in U=r·T form, compact FD + parity — COMPLETE (2026-05-15)

**Status: COMPLETE**, supersedes the original T-2 implementation above.

**Motivation**: the original T-2 used `standard_d1` FD with an i_r=0
NaN mask. This gave rate-2 convergence (the mask drops L'Hospital
cancellations from the 1/r terms in T_in_v). User requested:
1. Compact finite differences.
2. Parity conditions for the ball.

Direct retrofit failed because the 1/r factors in T_in_v themselves
generate inf/NaN at r=0 that no stencil choice can fix; need to clear
them symbolically. This led to the U-form formulation.

**Key formula** (derived in `scratch/derive_r2_div_in_U.py`, validated
in `scratch/validate_r2_div_in_U.py`):

Substituting `T_{σ_a, σ_b} = U_{σ_a, σ_b} / r` into the standard
(Div T)^{σ_1} formula and multiplying through by r² gives:

```
r² (Div T)^{σ_1} = polynomial(r) · U
                 + polynomial(r) · ∂_r U
                 + polynomial(r) · ∂_θ U
                 + cot(θ), m/sin(θ) couplings
```

— no 1/r factors. Similarly `U_in_v = r · T_in_v` is polynomial in r
(the 1/r from T_in_v cancels). Validated on 5 cases.

**Implementation pieces landed**:

1. `r2_div_tensor_spin_components(U_cov, m)` in tensor_reduce.py.
2. Walker emits U-form payload: `per_sigma_expr_in_U`, `U_placeholders`,
   `U_in_v`, `v_placeholders`.
3. `first_derivative_operators(bundle, parity=+1)` in operators.py:
   parity-aware compact-FD in ball mode, regular compact in shell,
   compact θ-d1.
4. `r2_vector_mass(bundle)`: (3N, 3N) diag(r²) block-diag mass with
   identity placeholders at i_r=0 in ball mode.
5. `_determine_equation_mass` returns `r2_vector_mass` for Div tasks
   — other equation terms automatically get left-multiplied.
6. `_dispatch_div_tensor_vector` rewritten for U-form; the 1/r mask
   becomes a defensive no-op.

**Convergence verdict**:

| Operator | Rate |
|---|---|
| Div(Grad(v)) at m=0 | ~5.6 |
| Div(Grad(v)) at m=+1 | ~5+ |
| Div(strain) at m=0 | ~2.86 |

**Open issue: Div(strain) rate-2.86 plateau.** Diagnosed but not yet
fixed. Truncation error τ peaks at corner (i_r=1, j_θ ≈ pole) at
~1e-3 (N=16), converging at rate ~2.85. Div(Grad) has the same
corner τ but its A⁻¹ damps it ~100× → rate-5 globally. Div(strain)'s
A⁻¹ only damps it ~4× → rate-2.85 globally. This is a property of
the PDE coercivity (Lap+Grad·Div weaker than pure Lap), not a stencil
bug. Tracked as `M7_corner_tau` below.

**Test thresholds tightened** in `tests/test_div_tensor_walker.py`:
- Div(Grad) N=32: 1e-5 → 5e-6 (measured 4.5e-7)
- Div(Grad) rate: ≥2.0 → ≥4.0 (measured 5.6)
- Div(strain) N=32: 1e-3 → 1e-4 (measured 5.1e-5)
- Div(strain) rate: ≥1.9 → ≥2.5 (measured 2.86)
- m=+1 N=32: 1e-4 → 1e-6 (measured 1.6e-7)


### M7_corner_tau: Div(strain) rate-4 via corner-τ damping (2026-05-15)

**Open issue**: rate-2.85 instead of rate-4 for Div(strain) on ball
manufactured tests. Diagnostic complete: localized
truncation error at (i_r=1, j_θ ≈ pole), propagated because Lap +
Grad·Div is less coercive than pure Lap.

**Proposed approaches** (not yet attempted):
1. **Augmented Lagrangian / iterated penalty**: regularize the
   Grad·Div piece to recover coercivity. Standard in mixed FEM, less
   so in FD.
2. **Higher-order corner stencil**: use a 5×5 or larger near-corner
   stencil with explicit pole-aware closure (would need a special-case
   row replacement for the (i_r=1, j_θ ∈ {0, 1, N-2, N-1}) cells).
3. **Better θ grid near poles**: Gauss-Legendre or stretched cell
   centers rather than uniform cell-centered. Would affect every
   operator, not just Div(strain).
4. **Accept it** for problems where 5e-5 at N=32 is sufficient (most
   Stokes-flow eigenvalue problems).

**RESOLVED in M7_v3 below**: the corner τ was reduced 17× by the
θ-parity-folded stencils, and the *separate* origin-regularity factor
bug (which dominated after the corner fix) was resolved by computing
the operator-specific factor symbolically. For ℓ=0-dominant solutions
the result is rate-3.7 at N=48 — see M7_v3 and the new M7_multi_ell_origin.


### M7 v3 (T-2 rev 2): θ-parity stencils + operator-aware origin (2026-05-15)

**Status: COMPLETE for ℓ=0-dominant solutions**; multi-ℓ case
documented under M7_multi_ell_origin.

**Motivation**: the M7 v2 cut delivered rate-5+ for Div(Grad) but only
rate-2.86 for Div(strain). Two separate bottlenecks were diagnosed and
fixed in this session.

#### Bottleneck 1: corner τ from cot(θ_pole) × θ-stencil error

At the (i_r=1, j_θ ≈ pole) corner, the operator coefficient cot(θ_0)
∼ N multiplies the θ-stencil error in `U[(-1,-1)]` (which has the
form `√2(-v^+/tan(θ) + ∂_θ v^+)` and discretizes with rate-4 truncation
at the pole on functions with v^+ ~ -sin(θ) near the pole). This
gives a corner-localized truncation error ~ N · N^{-4} = N^{-3}.

**Fix landed**: cell-centered parity-folded compact d1 stencils at
BOTH θ-endpoints (`compact_d1_cell_centered_parity` in stencils.py).
For m=0 axisymmetric:
- v^σ has θ-parity (-1)^{|σ|} at each pole.
- U_{σ_a,σ_b} has θ-parity (-1)^{|σ_a|+|σ_b|} at each pole.

The dispatcher applies the parity-correct θ stencil per-term (Stage 1
per v-component, Stage 2 per U-component). For m ≠ 0 the parity is
(-1)^{|m+σ|} but only m=0 is currently dispatched; m≠0 falls back to
the default D1_θ.

**Result**: corner τ reduced 17× (2e-3 → 1.2e-4 at N=16); revealed
that the i_r=0 origin row was the new bottleneck — Bottleneck 2.

#### Bottleneck 2: wrong origin regularity factor

The L'Hospital constraint at the origin was hardcoded as
`3 · v_z,rr(0) = src_z(0)` — correct for `Lap(v) = src` (and
incidentally for `Div(Grad(v)) = src` since Div(Grad) = Lap on
vectors), but **wrong for `Div(strain(v)) = src`** where the correct
relation at r=0 for ẑ-aligned ℓ=0 mode is `4 · v_z,rr(0) = src_z(0)`.

The factor comes from the operator's leading r→0 behavior for ẑ-aligned
ℓ=0 mode: `Div(strain v)(0) = (3 + 1) · v_z,rr(0) · ẑ` where the 3
comes from Lap and the +1 from Grad(Div(v)). Generally, for
`L = α Grad(v) + β Trans(Grad(v)) + γ (Div v) I`, the factor is
`3α + β + γ`.

**Fix landed**: walker derives the factor symbolically by substituting
`v = (r²/2)·ẑ` into the U-form and extracting the coefficient of
r²·cos(θ) in the σ=0 row (`_compute_origin_factor_for_div` in
walker.py). The factor is attached to task metadata, propagated through
`CompiledMatrices.origin_factors`, and consumed by
`_apply_ball_vector_origin_regularity_m0`.

**Result for Div(strain) on `v = (1-r²)ẑ`**:
- Before: rate 2.6, err 4.7e-5 at N=32, c_discrete diff = 4.7e-5.
- After: rate ~3.7, err 7.7e-7 at N=32, c_discrete diff = 7.7e-7.
- At N=48 (local run): err 1.7e-7, rate 3.68.

#### Files modified
- `spdee/backends/fd2/stencils.py`: added `compact_d1_cell_centered_parity`
- `spdee/backends/fd2/operators.py`: added `bundle.parity_d1_th(pl, pr)`
- `spdee/backends/fd2/compiler.py`: dispatcher applies per-term θ
  parities; `CompiledMatrices.origin_factors` propagates the factor
- `spdee/backends/fd2/compiled_problem.py`:
  `_apply_ball_vector_origin_regularity_m0(..., factor=3.0)` — default
  preserves all prior behavior
- `spdee/ir/walker.py`: `_compute_origin_factor_for_div`; `origin_factor`
  added to TASK_DIV_TENSOR_VECTOR metadata
- `tests/test_div_tensor_walker.py`: thresholds tightened to match
  measured rates

#### Test thresholds (tightened)
- Div(Grad) N=32: 1e-5 → 5e-6 (measured 9e-8)
- Div(Grad) rate: ≥2 → ≥4 (measured 4.01-4.11)
- Div(strain) N=32: 1e-3 → 5e-6 (measured 7.7e-7)
- Div(strain) rate: ≥1.9 → ≥3.3 (measured 3.59-3.68)
- m=+1 N=32: 1e-4 → 1e-6 (measured 1.6e-7)

#### Test count: 393 passing, 1 skipped. Confirmed by full pytest run.


### M7_multi_ell_origin: origin regularity for mixed ℓ vector solutions (2026-05-16)

**Open issue**: solutions with non-trivial angular structure on
Div(strain) plateau at rate ~2.18 (confirmed N=16, 24, 32, 48 on local
hardware).

#### Diagnosis

The current m=0 vector origin treatment has:
1. **One anchor row** at (σ=0, i_r=0, j=0):
   `factor · ⟨v_z,rr(0,·)⟩_{ℓ=0} = ⟨src_z(0,·)⟩_{ℓ=0}`
2. **3 N_θ - 1 Dirichlet ties** of all i_r=0 DOFs to the single anchor
   c = v^0(0, θ_0) / cos(θ_0), assuming `v(0) = c·ẑ` (the ℓ=0 Cartesian
   value).

For Lap, the anchor IS consistent for any ℓ-content of v at r=0
because Lap is "diagonal" in spherical harmonics at r=0 — each ℓ mode
of v contributes only to the same ℓ mode of Lap v at r=0.

For Div(strain) = Lap + Grad·Div, the Grad·Div piece **couples
different ℓ modes** at r=0. Specifically, ℓ=2 content of v_z
contributes 4/3 of its r²-coefficient to ℓ=0 of `Div(strain v)`(0,θ).
The single anchor can't represent this coupling.

**Concrete manifestation**: for `v_z = (1-r²) r² cos²θ` (= ℓ=0 + ℓ=2):
- ⟨v_z,rr(0,·)⟩_{ℓ=0} = 2/3
- ⟨src_z(0,·)⟩_{ℓ=0} = 4
- factor · 2/3 = 4 · 2/3 = 8/3 ≠ 4 — discrete inconsistent ⇒ rate-2
  in the anchor c value, propagating to rate-2 globally.

#### Proposed approach (not yet attempted)

Replace the single anchor row with a **block** of anchor rows, one per
significant ℓ mode at the origin:

For each ℓ ∈ {0, 2, 4, ...} (up to some cutoff ℓ_max):
```
sum over (σ_a, σ_b, m): A_{ℓ, (σ_a, σ_b, m)} · ⟨v^{σ_a,σ_b}_disc⟩_{ℓ}
    = ⟨src_z⟩_{ℓ}
```
where `A_{ℓ, (...)}` is the operator-specific coupling matrix (3 for
Lap diagonal, 4 for strain ℓ=0, 4/3 for strain ℓ=2 leakage into ℓ=0,
etc.).

The coupling matrix could be derived by the same symbolic substitution
used for the single factor — but parameterized by the test function's
ℓ content. E.g., for ℓ=2: substitute `v_z = (r²/2) Y_2(θ)` and read
off the resulting ⟨...⟩_{ℓ=2} pattern.

#### Workaround for users

If your physical solution has ℓ=0-dominant content (ẑ-aligned modes,
axisymmetric flows), rate ~3.7 applies. For mixed-ℓ solutions on ball
mode, expect rate-2 from origin coupling.

For shell mode the issue doesn't arise (no origin treatment, rate-5
across all tested ℓ content).

#### Priority

LOWER than M7 dogfooding (real-equation problem). Most physically
motivated eigenvalue problems on the ball will have ℓ-dominated
structure aligned with the magnetic / rotation axis (the ẑ direction
in our coordinates), which gives ℓ=0-like behavior at the origin in
the ẑ-aligned basis. The mixed-ℓ failure mode is more an artifact of
manufactured-solution testing than a real-world bottleneck.


### M7_m_nonzero_parity: θ-parity stencils for m ≠ 0 (2026-05-16)

**Open issue**: the new θ-parity-folded compact d1 stencils only
dispatch for m=0. For m ≠ 0, the dispatcher falls back to the default
(non-parity-folded) D1_θ.

The Cartesian smoothness argument generalizes: for m ≠ 0 axisymmetric
mode, v^σ at each pole has leading θ-power θ^{|m+σ|}, so its θ-parity
at the pole is (-1)^{|m+σ|}.

To extend: in `_dispatch_div_tensor_vector`, replace
```python
parity_v = (-1) ** abs(sigma_in_v) if m_mode == 0 else None
```
with
```python
parity_v = (-1) ** abs(m_mode + sigma_in_v)
```
and similarly for the U-component parity in Stage 2.

This wasn't done because (a) the m=+1 manufactured test was already
passing at rate ~5 without it (the bottleneck was elsewhere) and
(b) verifying the U-component parity rule for m ≠ 0 requires careful
spin-arithmetic that I haven't fully written out.

**Priority**: LOW. Only relevant if a real-world Div(strain) problem
at m ≠ 0 shows rate degradation. Pure Div(Grad) at m=+1 is already
rate-5+.


### M6_original: (superseded by the above — original notes preserved for reference) (2026-05-13)

**Where**: BC walker (`spdee/ir/walker.py`), BC dispatch in
`spdee/backends/fd2/compiled_problem.py`.

**Current state**: BC walker handles two patterns:
- Scalar Dirichlet (`p = g` or `p = 0`).
- Vector projection (`Dot(unit_vec, v) = g`), used for no-slip and
  impenetrable conditions.

**Missing**: tensor-derived BCs like stress-free angular boundary
conditions, which require taking the **angular** part of the
**radial** part of the stress tensor. Symbolically:

    BoundaryCondition(loc, Dot(r_hat, Dot(r_hat, sigma)) - something) = 0
    BoundaryCondition(loc, theta_hat . Dot(r_hat, sigma) - 0)       # = 0
    BoundaryCondition(loc, phi_hat   . Dot(r_hat, sigma) - 0)       # = 0

Where `sigma` is a stress tensor (a rank-2 expression built from the
unknown vector and a viscosity background). Implementing this requires:

1. Walker support for tensor expressions (currently rejected).
2. A path for tensor-derived BCs that produce row-vectors in the
   constraint matrix mixing multiple unknown columns and possibly
   spatial-derivative stencil weights.

**When to revisit**: when MHD eigenproblems with realistic free-slip
boundaries enter scope. Most academic settings allow no-slip as an
acceptable approximation initially, so this isn't blocking.

**Possible refinements**:
- Introduce intermediate aliases at the IR level: `sigma = ...` as
  a derived tensor, then BCs reference it.
- Extend the BC walker dispatcher to recognize specific compositions
  of `Dot` with `Grad(v)` / etc., emit appropriate row patterns.
- Likely needs state augmentation (introduce auxiliary tensor field).


### M7: Persist operator matrices to disk for MPI shift-and-invert sweeps (2026-05-14)

**Where**: Future Stage 4 eigenproblem infrastructure. Currently the
matrices live only in memory and the in-memory LU cache
(`scratch/poisson_dirichlet.py::LUCache`) only helps within a single
process.

**Motivation**: For Stage 4, MPI processes will sweep different shift
values `σ` in shift-invert eigenproblem iterations `(A - σ B)^{-1} v`.
Each shift produces a different LU factorization, but `A` and `B`
themselves are identical across MPI ranks. Building those matrices is
non-trivial at large N; recomputing them per rank is wasteful.

The plan:
- Rank 0 builds `(A, B)` and serializes them (scipy.sparse format or
  similar) to disk.
- Other ranks read the matrices from disk and only build their own
  shifted LU.
- Optionally cache LUs to disk too, though those are typically larger
  than the matrices themselves.

**When to revisit**: Stage 4 eigenproblems with MPI parallelism.

**Possible refinements**:
- A `MatrixStore` class with `save(matrices_dict, path)` and
  `load(path)` methods. Format: a single HDF5 file (h5py) per
  (problem-spec, geometry, m, stencil) tuple. Inside: CSR data arrays
  for A_op, B_op, and any per-equation mass matrices.
- Cache key includes a hash of (system definition, grid resolution,
  stencil, geometry) so reloads don't accidentally pick up stale
  matrices.
- Helper: read matrix once on rank 0, broadcast via MPI.Bcast for
  the small extras (grid metadata, layout); read the bulk via parallel
  HDF5 for large matrices. Decide which is better empirically.


### M8: Composed-operator walker patterns — COMPLETE (2026-05-18 ... 2026-05-19)

**Where**: `spdee/ir/walker.py`, `spdee/ir/tasks.py`,
`spdee/backends/fd2/{compiler,grad_div_sparse,dot_bg_vec_tensor_unk_sparse}.py`.

**Original concern**: state augmentation for composed operators
(per M2) was the easy fallback but scales poorly at production
resolutions because each aux scalar adds `N_r·N_θ` extra rows and
columns to the matrix. The user's hard production constraint is
**no state augmentation** for the dominant operators in the Bekki
inertial-mode replication.

**Resolution**: dispatched the three production-critical composed
operators directly via dedicated walker patterns + sparse builders.
Each is a single 3N×3N (or 3N×N, etc.) sparse block per equation
row, with no aux scalar introduced:

  * `TASK_GRAD_DIV_VECTOR` — `Grad(Div(unk_vec))`. r-power 2.
    Builder: `build_r2_grad_div_vector_block` composes the
    `Grad(scalar)` formula on top of the `Div(vec)→scalar` formula
    symbolically, using ACTION_D_RR / D_TT / D_RT stencils. Origin
    factor derived for ẑ-aligned ℓ=0.
  * `TASK_DOT_BG_VEC_TENSOR_UNK` — `Dot(bg_vec, tensor_expr(unk_vec))`.
    r-power 1. Builder:
    `build_r_dot_bg_vec_tensor_unk_block` reduces the tensor
    expression via the IR's `reduce_tensor_expression` to covariant
    spin components, then applies the V19 contraction
    `(Dot(a, T))^σ_out = Σ_σ_a a^σ_a · T_{σ_a, -σ_out}`.
  * `TASK_BG_VEC_TIMES_DIV_UNK_VEC` — `bg_vec * Div(unk_vec)`.
    r-power 1. Trivial composition: reuses the existing
    `build_div_vector_scalar_block` to compute `r·Div(v)` and
    multiplies by the bg vector's spin components on the grid.

These three together (plus the existing `Lap(v)`) cover the full
Bekki §2.3 compressible viscous force
    (1/ρ_0) ∇·(ρ_0 ν T)  =  ν [ ∇²v + (1/3)∇(∇·v)
                              + ∇ln ρ_0 · (∇v + (∇v)^T)
                              − (2/3)(∇·v) ∇ln ρ_0 ]
with zero aux unknowns.

**Future generalization** (if needed): the pattern that composes
`Grad ∘ Div` could in principle be extended to `Lap ∘ Div`,
`Curl ∘ Div`, etc. Each would need its own walker pattern and
backend builder. For now we only build the compositions Bekki
actually needs.

### M9: Neumann scalar BC walker — COMPLETE (2026-05-20)

**Where**: `spdee/ir/walker.py` BC dispatch path,
`spdee/backends/fd2/compiled_problem.py` BC application path,
`spdee/ir/tasks.py` BC task vocabulary.

**Original concern**: Bekki uses `∂_r s_1 = 0` (Neumann) on the
entropy field at radial boundaries. spdee's BC walker only
supported Dirichlet (`s = 0`) and impenetrable/stress-free vector
BCs. Step 4 used Dirichlet `s_1 = 0` as a placeholder.

**Resolution**: added `BC_NEUMANN_SCALAR_RADIAL` task kind. The
walker recognizes both `Dot(r_hat, Grad(unknown_scalar))` and the
inhomogeneous form via the existing `Sum` parser, accepting either
argument order. The backend installs the BC via row replacement
using a one-sided FD stencil of width `order_r + 1`, computed by
the existing `_lagrange_derivative_weights` (so the BC stays
4th-order accurate on non-uniform grids).

The host-row collision check treats Neumann and Dirichlet on the
same scalar at the same boundary as a collision — they share the
same boundary row and one would silently overwrite the other.

**Validation**:

* Manufactured-solution BVP at multiple resolutions
  (`tests/test_neumann_scalar_bc.py::TestNeumannBVP::test_rate_4_convergence`):
  clean rate-4 convergence (4.20 → 4.06) on `sin(2r)` with mixed
  Dirichlet+Neumann BCs.
* Pattern-recognition tests for all four walker entry points (bare
  Dot, reversed args, inhomogeneous with bg-on-RHS or bg-on-LHS).
* Collision-detection test.
* Bekki Step 4 integration: with `entropy_bc='neumann'` vs
  `entropy_bc='dirichlet'`, the n=0 sectoral Rossby mode produces
  IDENTICAL eigenvalues to roundoff (Δω, Δγ ~ 1e-14). This
  empirically confirms the docstring claim that the entropy BC
  choice is irrelevant for Rossby modes (which are toroidal-
  velocity dominated and have s_1 ≈ 0 throughout). For entropy-
  dominated modes the BCs would differ.

The Step 4 builder gained an `entropy_bc='dirichlet'|'neumann'`
parameter; the default stays `'dirichlet'` for backward compat
with the existing test suite, with `'neumann'` available for
production Bekki Fig 22a comparisons.

### M10: Walker/dispatcher boilerplate consolidation — COMPLETE (2026-05-20)

**Where**: `spdee/ir/walker.py`, `spdee/backends/fd2/compiler.py`.

**Concern**: each new operator added across the project required
duplicating ~60 lines of dispatcher boilerplate (assert TARGET_A,
build, mass-conversion, coefficient, sign, accumulate) and ~12 lines
of walker emit boilerplate (build the full OperatorTask with target/
equation_name/sign/coefficient/metadata['m']). The 19 task kinds
collectively had ~700 lines of nearly-identical dispatcher code
and ~200 lines of nearly-identical emit code.

**Resolution**: two helpers consolidate the boilerplate.

1. **`compiler._finalize_contribution(contribution, task, A_eq, grid,
   *, output_kind, pointwise)`** runs the standard four-step post-
   build pipeline (mass alignment vs A_eq left-multiply, coefficient
   application, sign, return). 13 of 14 differential/composed-op
   dispatchers now use it. Identity (target A or B based on Dt
   context) and Source (writes to b_total) dispatchers are
   intentionally outside the framework.

2. **`_WalkContext.emit_task(kind, unknown_name, *, target=TARGET_A,
   source_field=None, extra_metadata=None, coefficient_override=None)`**
   builds the OperatorTask filling the boilerplate fields from
   ctx. All 18 walker emit sites use it.

**Effect**: 261 lines cut from compiler.py (2229 → 1968) and 37
lines from walker.py (1847 → 1810). Adding a new operator now
requires ~20 lines of dispatcher (was ~60) and ~5-10 lines of
walker emit (was ~15-20). The dispatch trees in both files
remain explicit (no registry indirection) which keeps the
list of supported operators discoverable in one place.

Verified: 693 tests + 1 skipped, no regressions.

### M11: Richardson extrapolation as the Bekki-comparison primitive (2026-05-20)

**Where**: `tests/test_solar_shell.py::TestCompressibleStep4::test_richardson_extrapolation_to_continuum`.

**Concern**: Bekki (2022) uses a staggered Arakawa-C / MAC-style
spherical grid (``v_r`` on r-faces, ``v_θ`` on θ-faces, scalars and
``v_φ`` at cell centers); spdee uses a colocated grid (all fields at
the same nodes). Both layouts are 4th-order accurate and converge
to the same continuum eigenvalues, but their leading finite-N
correction coefficients differ. A direct N_r=N value comparison
between the two codes is therefore expected to disagree at finite
N even when both are correct.

**Resolution**: Compare at the CONTINUUM limit, not at a chosen N.
The Richardson-extrapolation test fits ``ω(N_r) = ω_∞ + C·N_r^{-p}``
to a ladder of well-resolved N_r values at a fixed (m, Ek), reports
``ω_∞`` and the convergence exponent ``p``, and asserts that ``p`` is
consistent with 4 (the formal order of the FD scheme). The extra-
polated ``ω_∞`` is the layout-independent quantity that should agree
between spdee and Bekki to the precision of each code's truncation
error.

Sandbox-validated continuum value at m=3, Ek=3e-3, Neumann entropy
BC: ``ω_∞ = -0.464360 ± 3×10⁻⁵`` (p = 4.25, fit on ladder
N_r ∈ {24, 32, 48, 60}). The sandbox can extend this ladder up to
about N_r=72 within the ``slow``-test budget; production hardware
can push to N_r=128 or higher and lower Ek toward Bekki's Ek=7×10⁻⁵
for direct Fig 22a comparison.

**Future use**: the same pattern is the right way to compare any
discrete spectrum from spdee against any reference whose discrete
layout differs from spdee's — extract continuum limits from both
sides and compare those.

**Possible refinements**:

* Add an `m`-sweep version: for each m ∈ {1, …, 16}, produce
  ``ω_∞(m)`` from a small Richardson ladder. The output is a table
  directly comparable to Bekki Table or Fig 22a.
* Track γ alongside ω. The damping rate has a smaller convergence
  exponent (sandbox shows p ≈ 3 for γ at this Ek), indicating
  boundary-layer effects in the damping that aren't yet fully
  characterized. Production-scale data would clarify whether this
  is genuine (Ekman-layer contribution to γ that the rule
  ``δ_E ≥ 5·Δr`` underestimates) or numerical artifact.

### M12: Staggered I/O and layout-shift quantification — COMPLETE (2026-05-20)

**Where**: `spdee/backends/fd2/staggered_io.py` (new module),
`tests/test_staggered_io.py` (new test file).

**Context**: Bekki (2022) uses an Arakawa-C / MAC-style staggered
spherical grid. spdee uses a colocated grid. The question for any
direct comparison is: does the layout matter for the comparison?

**Resolution**: Two complementary pieces:

(a) ``staggered_io.py`` provides arbitrary-order Lagrange
interpolators between spdee's colocated grid and Bekki's staggered
(r-face / θ-face / cell-center) grids. The module exposes
``to_staggered`` and ``from_staggered`` for round-trip
conversion of solutions; this is a comparison utility only,
unrelated to the solver pipeline. Pole values for ``v_θ`` are
forced to 0 by physical-regularity (the coordinate singularity at
the pole is not extrapolated through).

The interpolators are degree-``order`` Lagrange polynomials on
local stencils of width ``order + 1``. Note that the polynomial
degree gives **rate-``order+1``** convergence in the smooth-target
case (e.g., ``order=4`` gives Δ⁵ scaling) — one power higher than
the analogous derivative weights, because interpolation only needs
to fit values, not approximate derivatives.

(b) The layout-shift sensitivity test (``TestLayoutShiftSensitivity``)
characterizes how much the spdee Step 4 Rossby eigenvalue depends
on detailed grid alignment beyond the bulk Δr trend. It solves at
adjacent N_r ∈ {40, 41, 48, 49} (where the cell centers all shift
slightly between configurations) and measures the eigenvalue
spread. Sandbox result: ``1.3×10⁻⁴`` spread, which is 0.03% of
|ω| at this Δr. At production scale (N_r=128) the layout residual
extrapolates to ~5×10⁻⁷.

**Implication for Bekki Fig 22a comparison**: the colocated-vs-
staggered layout difference contributes only ~5×10⁻⁵ at moderate
resolution and disappears at the continuum limit. Comparison via
Richardson extrapolation (M11) is therefore the right tool —
spdee's continuum eigenvalue is layout-independent, and Bekki's
published value is its own continuum estimate. No staggered backend
is needed.

**Possible refinements**:

* When/if a digitized Bekki snapshot is available, compute
  panel-by-panel mode-shape comparison after converting spdee's
  eigenmode via ``to_staggered``. Currently the eigenmode
  round-trip test confirms the conversion is faithful (within 3%
  relative ℓ² for the m=3 Rossby mode at N_r=48) but no external
  reference is plugged in yet.
* The same interpolators would be used to seed spdee from a
  Bekki snapshot if time-stepping comparison were ever wanted.

### M13: Aux-free MHD — COMPLETE (2026-05-20)

**Status**: implemented and validated.

The two MHD composed operators that historically required state
augmentation are now dispatched as **direct matrix products of
elementary Curl and Cross blocks** with no auxiliary unknowns:

  `Cross(Curl(unknown_vec), bg_vec)`  →  `X_R(bg) @ C`  (Lorentz force)
  `Curl(Cross(unknown_vec, bg_vec))`  →  `C @ X_R(bg)`  (induction)

This collapses the state from N²·(7 + N_aux·3) back to N²·7 for the
hydrodynamic Bekki setup, or N²·8 when a magnetic perturbation `b` is
also introduced (no aux for the composed operators on `b` either).

**Implementation pieces**:

* `spdee/backends/fd2/curl_cross_sparse.py`:
  `build_cross_curl_unk_bg_block`, `build_curl_cross_unk_bg_block`,
  `_make_curl_walker_meta` — composed-block builders.
* `spdee/ir/tasks.py`: `TASK_CROSS_CURL_UNK_BG`, `TASK_CURL_CROSS_UNK_BG`.
* `spdee/ir/walker.py`: pattern recognition in `_emit_curl` and
  `_emit_cross` for the composed patterns; emitted before the
  bare-FieldRef fallback. Coefficient-extraction recursion through
  nested `ScalarMultiple` and `Negation`.
* `spdee/ir/augment.py`: `prefer_aux_free` flag on
  `augment_equation_system`, default `True`. Aux-augmentation
  only runs for patterns the walker can't otherwise handle.
* `spdee/backends/fd2/compiler.py`: `_dispatch_cross_curl_unk_bg`,
  `_dispatch_curl_cross_unk_bg`; r-power table extended.
* `spdee/backends/fd2/compiled_problem.py`: `prefer_aux_free`
  parameter passes through to the augmentation pass.

**Validation**:

* `tests/test_mhd_composed_blocks.py` (8 tests): pointwise rate-4
  accuracy of each composed block against analytic answers.
* `tests/test_mhd_aux_free.py` (6 tests): bit-for-bit equivalence
  between aux-augmented and aux-free paths on the same problem
  across multiple B₀ amplitudes; eigenvalues agree to ~3e-11.

**Memory savings at production scale**: at (256, 128) with the full
MHD Bekki system (8 unknowns), the aux-free state is 262 k DOF
vs ~ 590 k DOF for the aux-augmented equivalent — a factor of ~ 2.3.
LU memory scales as DOF² × bandwidth, so the savings are roughly
6× in matrix factorization memory.

### M14: Bekki + dipolar B₀ application — COMPLETE (2026-05-20)

**Status**: implemented and validated.

`spdee/applications/solar_eigenmodes.py` provides:

* `make_dipolar_B0(name, r_inner, r_outer, amplitude_ratio)` —
  divergence-free poloidal axisymmetric field from stream function
  ψ(r,θ) = f(r) sin²θ with f(r) chosen so |B_θ(equator)| has
  the requested ratio between r_inner and r_outer. Default ratio
  30 000 = 30 kG → 1 G.
* `make_dipolar_B0_curl(...)` — Curl(B₀) precomputed as a separate
  background. Purely toroidal for the dipolar profile.
* `make_solar_shell_system(solar, m, Ek, Le, Pm, ...)` — extends
  Bekki Step 4 with magnetic unknown b, Lorentz force, induction
  equation, perfect-conductor BCs.

**Sweep driver**: `scripts/sweep_solar_mhd_Le.py` — Le-sweep at
fixed m, fixed grid. Same infrastructure as `sweep_solar_ball.py`
(spawn workers, threadpoolctl caps, heartbeat logging, per-stage
profiling, resumable manifest).

**Validation** (`tests/test_solar_eigenmodes.py`, 9 tests):

* Field profile: amplitude ratio exact, ∇·B = 0 analytically at
  sample points (< 1e-10), B_φ identically zero, Curl(B₀) purely
  toroidal.
* Le=0 reduces to hydro: every Bekki Step 4 eigenvalue appears in
  the MHD spectrum to ~1e-14.
* Le² scaling holds: doubling Le between 0.05 and 0.10 quadruples
  Δω (ratio 4.0 ± 0.5).

**Empirical scaling** at N=48×48, Ek=3e-3, m=3:

| Le     | ω           | γ          | |Δω|/Le² | |Δγ|/Le² |
|--------|-------------|------------|----------|----------|
| 0      | -0.46444    | -0.29423   | —        | —        |
| 1e-3   | -0.46444    | -0.29423   | 0.172    | 0.175    |
| 1e-2   | -0.46443    | -0.29422   | 0.172    | 0.175    |
| 3e-2   | -0.46429    | -0.29408   | 0.173    | 0.175    |
| 1e-1   | -0.46270    | -0.29246   | 0.174    | 0.177    |

The Δω, Δγ coefficients are positive: the magnetic field reduces
both the retrograde frequency (less negative ω) and the damping
(less negative γ).

**Deferred (not blocking)**:

* **Potential-field BC for b** at r_outer. Currently perfect-
  conductor (b = 0 at the wall). Potential-field matching requires
  spherical-harmonic expansion of b at the boundary to match a
  curl-free vacuum extension. Documented as TODO in `solar_eigenmodes.py`
  module docstring. The perfect-conductor case provides a
  conservative bound on the magnetic effect; potential-field will
  give slightly different absolute numbers but the same scaling.
* **Production-resolution sweep**: infrastructure is ready;
  recommended invocation is documented in
  `CHECKPOINT_2026-05-20_solar_eigenmodes_complete.md`. Estimated cost
  is ~ 4 hr on a 16-core node for 9 Le values × 8 m values at
  (256, 128).

### M15: Sweep-result analyzer — COMPLETE (2026-05-20)

**Status**: implemented and tested.

`scripts/analyze_sweep.py` auto-detects sweep type (ball-by-m or
Le-by-amplitude) and produces:

* Text summary on stdout (tab-aligned).
* Markdown summary file (`analysis_summary.md`) in the sweep dir.
* PNG plot (`summary.png`) — for ball sweep: 2 panels of ω(m) and γ(m)
  for each cell; for Le sweep: ω(Le) for each m plus a log-log |Δω| vs
  Le panel with a 0.17·Le² reference line.

For ball sweeps with multiple cells, the analyzer compares **every pair
of cells** rather than just the first two, labels each pair by which
grid axis it's resolving (N_r, N_θ, or both), and flags mode jumps
where |Δω| > 1e-3.

Validation: 15 unit tests in `tests/test_analyze_sweep.py` use
synthetic record dicts (no I/O, no compute) to verify
`_detect_sweep_kind`, `_analyze_ball`, `_analyze_le`, mode-jump
detection, multi-cell pair generation, the Le² scaling-law extraction,
and that all four formatters (text + markdown × ball + Le) produce
non-trivial output without crashing.

### M16: Spectrum scan (Figure-2-style) — COMPLETE (2026-05-20)

**Status**: implemented and tested.

The Le-sweep (M14) does one targeted solve per (m, Le). To reproduce
a Bekki-Figure-2-style spectrum we instead scan a fine grid of
oscillatory-frequency targets across an inertial band and union the
per-target solves.

**EVP refactor** (`spdee/backends/fd2/compiled_problem.py`): factored
`solve_evp` into reusable pieces so a scan assembles the σ-independent
(K, M) once and reuses them across thousands of shift targets:

* `assemble_evp_matrices(mass_equations=None)` → `(K_csr, M_csr)` with
  origin regularity, BC row replacement, and gauge constraints applied
  to K, and M built and row-zeroed to match.
* `_build_mass_matrix(...)`, `_resolve_mass_terms(...)` — shared
  helpers extracted from the monolithic solve_evp.
* `solve_evp_from_matrices(K, M, σ, k)` — static method, the
  per-target inner loop: form (K − σM), LU, ARPACK, map shifted-
  inverse spectrum back. Bit-for-bit equal to monolithic solve_evp
  (verified to relative 1e-12; the residual is ARPACK random-start
  roundoff, not a logic difference).

The monolithic `solve_evp` was subsequently refactored (2026-05-21) to
*delegate* to these helpers: it calls `assemble_evp_matrices` then, for
a shift-invert solve, `solve_evp_from_matrices` (or `scipy.eigs`
directly for the σ=None direct path), wrapping the result in
`EvpResult`. Behavior is identical — verified by
`tests/test_spectrum_scan.py::test_eigenvalues_match_monolithic` and
the full EVP test suite (40 core EVP tests + 32 MHD tests) — but the
assembly/solve logic now lives in exactly one place rather than being
duplicated between `solve_evp` and the helpers.

**Scan driver** (`scripts/scan_solar_mhd_spectrum.py`): MPI-parallel,
mirroring the reference Dedalus script's two-level structure:
* Outer loop over (m, Le): rank 0 assembles (K, M) and saves to disk
  via `scipy.sparse.save_npz`; all ranks load the shared matrices.
* Inner loop over targets: the target grid (default 5001 points over
  ω ∈ [−2, 2]) is divided round-robin across ranks; each rank does
  LU + ARPACK per target for `k` (default 64) eigenvalues.
* Convention: σ = γ_t − i·ω_t with fixed γ_t = −0.001 (small damping
  offset), scanning the physical oscillatory frequency ω_t. This
  matches spdee's λ = −iω ansatz (ω = −Im λ, γ = −Re λ).
* Output per cell: `raw_scan.npz` (per-target eigenvalues, padded to
  (n_targets, k)) and `spectrum.npz` (deduped unique modes). Dedup
  rounds (Re, Im) to a tolerance (default 1e-6) and keeps one
  representative per bucket.
* Falls back to serial if mpi4py is unavailable or launched without
  mpiexec.

**Plotter** (`scripts/plot_spectrum.py`): collects all `spectrum.npz`
cells under a directory and renders:
* `dispersion.png`: ω vs m scatter, one color per Le, marker opacity
  ∝ weak damping, with the −2/(m+1) Rossby asymptote.
* `complex_m{NN}.png` (with `--complex-plane --m N`): ω vs γ scatter
  at fixed m comparing Le values.

**Validation** (`tests/test_spectrum_scan.py`, 11 tests):
* Factored path reproduces monolithic solve_evp (relative 1e-12)
  across multiple shifts and from one assembly.
* (K, M) round-trip through save_npz/load_npz unchanged; solving from
  loaded matrices matches solving from assembled.
* Dedup collapses duplicates, keeps distinct modes, drops NaN,
  sorts monotonically.
* σ = γ − iω convention recovers a known eigenvalue when targeted.

**Cost note**: the scan does (n_targets) LU factorizations + ARPACK
runs per (m, Le). Each LU is the dominant cost and is σ-dependent,
so it cannot be amortized across targets — the matrix-reuse only saves
the *assembly*, not the factorization. At N=128² with k=64 the per-
target solve is the binding cost; budget accordingly and use MPI
breadth across targets. At the production scale this is exactly the
Dedalus script's cost model.


### M17: Choices behind keywords that default to the conservative option (2026-09-17)

**Where**: `pole_closure` on the staggered core, `impenetrability_host`
on the collocated core, the latitudinal means of the solar background.

- `pole_closure=True` on the staggered core adds the explicit pole share
  of the strain's theta-phi component, which keeps the pole-adjacent
  azimuthal velocity ring second order at |m| >= 2 (without it: order 2
  at 24 cells, order 1 at 48).  It moves the pole rows of the viscous
  block away from the certified recipe, so the entrywise certification
  would change there while the Rossby mode and the dense spectrum do not.
  Default off.
- `impenetrability_host=` on the collocated core: `"constraint"` hosts
  the impenetrable-wall condition on the continuity wall row and kills
  the pressure checkerboard at m = 0, but leaves the inviscid operator
  exactly singular at m != 0 (the normal momentum row's FD matrix has
  constants in its kernel) and costs accuracy on the Zhang targets; the
  default `"auto"` uses it at m = 0 only.
- The sin(theta)-weighted latitudinal means in the solar background move
  the detected tachocline from 0.686 to 0.709 and the blended
  inner-boundary rotation rate at the shell base from 450.6 to 442.7 nHz;
  whether that is the physics wanted for Model S shell runs with the
  blend on is a modelling decision, not a numerical one.

**Revisit when**: the staggered viscous pole rows are re-certified, or
the collocated inviscid m != 0 singularity gets a proper host.
