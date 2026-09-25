# Operator Composition (State B)

All equations are compiled through a single unified path. Each equation
may contain any number of differential and pointwise operators acting on
possibly-different unknowns, subject to output-kind consistency (all
vector-output or all scalar-output).

This handles uniformly:
- Single-operator equations: `Lap(v) = src`, `Curl(v) = src`, etc.
- Multi-operator equations: `Lap(v) + Curl(v) = g`,
  `Lap(v) + Grad(p) + 2·Cross(Ω, v) = g`, `a·Lap(v) - b·Curl(v) = g`.
- Coupled systems: e.g., the inertial-wave momentum + continuity
  pair, with optional gauge constraints on pressure-like unknowns.

## Design

### Common r-power mass

Each differential operator has a **natural r-power**: the smallest
non-negative integer `n_op` such that `diag(r^n_op) · op(v)` is
polynomial in r near r=0 (no 1/r factors).

| Operator | Natural r-power |
|----------|-----------------|
| Identity, Cross, Source | 0 |
| `Dot(bg_tensor, unk_vec)`, `Dot(bg_vec, unk_vec)→scalar`, `bg_vec * unk_scalar` | 0 |
| Grad, Curl, Div(vector)→scalar | 1 |
| `Dot(bg_vec, Grad(unk_vec))` (advection) | 1 |
| `Dot(bg_vec, tensor_expr(unk_vec))` (generalized contraction) | 1 |
| `bg_vec * Div(unk_vec)` (trace-piece composition) | 1 |
| `Cross(Curl(unk_vec), bg_vec)` (aux-free Lorentz) | 1 |
| `Curl(Cross(unk_vec, bg_vec))` (aux-free induction) | 1 |
| Lap(scalar), Lap(vector), Div(tensor)→vector | 2 |
| `Grad(Div(unk_vec))` (bulk-divergence composition) | 2 |

For each equation, the compiler picks the **maximum natural r-power**
across all differential tasks, denoted `p_eq`. The equation's mass is
then `diag(r^p_eq)` on each output spin block (with i_r=0 identity
placeholders in ball mode).

Each differential operator's contribution is converted from its
natural mass `diag(r^n_op · k_op)` to the common mass `diag(r^p_eq)`
via a diagonal scaling:
```
contribution_at_p_eq = diag(r^{p_eq - n_op} / k_op) · contribution_natural
```
where `k_op` is the operator's constant factor (1 for most, `√2` for Grad).
Pointwise operators (Cross, Identity, Source) use the equation mass
directly via the existing `A_eq @ contribution` path.

### Sparse-banded operator builders

All operators are implemented as **sparse-banded builders** that take
the bundle, the unknown's spin placeholders, and azimuthal mode m, and
return a sparse matrix in their natural r-power form:

- `spdee/backends/fd2/lap_sparse.py`:
  - `build_r2_lap_scalar_block(bundle, m, p_unknown_name)` →
    `(N, N)` block such that `diag(r²) · Lap(p) = block · p_state`.
  - `build_r2_lap_vector_block(walker_meta, bundle, m)` →
    `(3N, 3N)` block such that `r2_vector_mass · Lap(v) = block · v_state`.

- `spdee/backends/fd2/curl_cross_sparse.py`: Curl and Cross builders,
  plus the aux-free composed MHD builders
  `build_cross_curl_unk_bg_block` (Lorentz force, `X_R(bg) @ C`) and
  `build_curl_cross_unk_bg_block` (induction, `C @ X_R(bg)`). The
  composed forms are literal matrix products of the elementary
  blocks, so they automatically inherit the same r-power and equation-
  mass machinery — no special handling in the compiler.
- `spdee/backends/fd2/grad_div_sparse.py`: Grad(scalar), Div(vector)→scalar,
  and Grad(Div(unk_vec)) builders. The latter is a composed
  second-order operator that avoids state augmentation for `∇(∇·v)`.
- `spdee/backends/fd2/div_tensor_sparse.py`: Div(tensor)→vector builder
  + the shared `_decompose_v_expr` / `_coef_to_diag` symbolic helpers
  and the `ACTION_*` constants.
- `spdee/backends/fd2/advection_sparse.py`: `Dot(bg_vec, Grad(unk_vec))`
  advection-style builder.
- `spdee/backends/fd2/dot_bg_vec_tensor_unk_sparse.py`:
  `Dot(bg_vec, tensor_expr(unk_vec))` builder. Reduces the tensor
  expression to covariant spin components via the IR
  `reduce_tensor_expression`, then applies the V19 contraction
  `(Dot(a, T))^σ_out = Σ_σ_a a^σ_a · T_{σ_a, -σ_out}`.

All use **standard 4th-order centered FD** throughout, yielding rate-4
convergence on smooth inputs in both shell and ball geometry.

### Origin-regularity factor accumulation

Ball-mode origin regularity installs an L'Hospital-style row at
i_r=0 of the form `factor · v_z''(0) = source_z(0)` for ẑ-aligned
ℓ=0 mode. The factor is the **sign-weighted, coefficient-weighted sum**
of contributions from each differential task acting on the unknown:

```
factor = sum_i  sign_i · coef_i(r=0, θ-averaged) · base_factor_i
```

where `base_factor_i` is:
- 3 for Lap (scalar or vector)
- 4 (or whatever the walker computes) for Div(tensor); set via
  `task.metadata['origin_factor']`
- 0 for Curl, Cross (their ẑ-projection vanishes on ℓ=0)

This correctly handles cases like `2·Lap(v) - Curl(v) = g` where the
factor should be `6 = 2·3 + 0`, not `3` (Lap default).

The `_coefficient_value_at_origin` helper evaluates the coefficient on
the i_r=0 grid row and averages over θ — appropriate for the ℓ=0
projection of Cartesian-smooth coefficients.

## Validation

Per-operator convergence lives in the generated matrix,
`docs/OPERATOR_COVERAGE.md`: one manufactured-solution boundary-value
problem per (operator, core, geometry, azimuthal order), with the
measured order or the core's own refusal. `scripts/operator_coverage.py`
regenerates it and `tests/test_operator_coverage.py` asserts it.
Multi-operator equations are covered there through the carrier form
`Lap(u) + Op(u) = f`, which is exactly the composition path.

The pressure p has the colocated-FD pressure checkerboard (LBB
instability). Standard fixes: staggered grids, pressure stabilization,
or toroidal-poloidal decomposition.

## Known limitations

1. **LBB checkerboard for pressure**: the primitive-variable
   formulation with colocated FD has the classical pressure
   checkerboard (`tests/test_checkerboard_probe.py`). The Bekki replication (Stage T-2+) avoids this by
   using the compressible continuity equation directly (with ρ_1
   as an explicit unknown) — no pressure unknown is needed.
   Workarounds for incompressible Stokes-like systems: pressure
   stabilization, gauge elimination (already supported via
   ``gauge_unknowns``), or toroidal-poloidal decomposition.

2. **Coefficient evaluation at origin**: The
   `_coefficient_value_at_origin` helper θ-averages the coefficient
   at r=0 for the ℓ=0 projection. This is correct for
   Cartesian-smooth coefficients but may need refinement for
   coefficients with strong θ-structure at the origin.
