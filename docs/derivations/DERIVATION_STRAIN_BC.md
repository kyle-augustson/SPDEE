# Derivation note: stress-free angular BC

**Date:** 2026-05-14
**Status:** Math derived and symbolically verified. Implementation pending (next session).
**Scope:** Two angular-stress BC formulas for the strain-rate tensor at an impenetrable wall.

## Context

Impenetrable stress-free wall at r = R_outer (or r = R_inner) imposes three scalar BCs:

1. **Impenetrability:** v_r = 0 (i.e., v^0 = 0 since v_r = v^0 in V19 spin basis).
2. **Stress-free angular:** the angular components of the strain-rate traction vanish:
   - `(E[v])^{+1, 0} = 0`
   - `(E[v])^{-1, 0} = 0`
   
   where `E[v] := Grad(v) + Grad(v)^T` is the symmetric strain-rate tensor.

The trace correction `-(2/3) I Div(v)` (compressible case) only contributes
to the diagonal `(σ, σ)` components of E, NOT to the σ ≠ σ' off-diagonal
components. So whether the equations are incompressible or compressible,
the angular-stress BCs are the same.

The viscosity coefficient `μ(r, θ)` (or any other multiplicative
background) is handled separately — inside equations rather than BCs.
The BC math is purely the geometric strain-rate tensor.

## Math

For a vector field v in azimuthal mode m (with e^{i m φ} factored out)
written in V19 spin basis (σ ∈ {-1, 0, +1}):

```
E^{+1, 0} = ∂_r v^-  -  v^- / r  +  (1/(√2 r)) ∂_θ v^0  +  (m / (√2 r sin θ)) v^0
E^{-1, 0} = ∂_r v^+  -  v^+ / r  +  (1/(√2 r)) ∂_θ v^0  -  (m / (√2 r sin θ)) v^0
```

**Structural notes:**

- `E^{+1, 0}` couples `v^-` (via radial derivative and pointwise) with `v^0`
  (via angular operator). It does NOT involve `v^+`.
- `E^{-1, 0}` couples `v^+` (via radial derivative and pointwise) with `v^0`
  (via angular operator). It does NOT involve `v^-`.
- The angular operator on `v^0` is the spin-raising / spin-lowering
  operator: `(1/√2)(∂_θ ± m/sin θ)`.

**m → -m symmetry:** the formulas are related by `m ↔ -m` and `v^+ ↔ v^-`,
consistent with the spin-basis conjugate-symmetry of azimuthal modes.

## Verification

Verified symbolically in `scratch/verify_strain_bc.py`:

- **Rigid-body rotation** (`v = Ω r sin θ ê_φ`, m=0): both BCs evaluate to
  exactly zero. ✓
- **Pure-θ flow** `v_θ = f(r)` at m=0: both BCs evaluate to
  `(f' − f/r)/√2`, which vanishes precisely when `f(r) = c·r` — the
  rigid-body case in cylindrical-like coordinates. ✓
- Structure preserved at m≠0. ✓

Derivation source: `scratch/derive_strain_bc.py` (computes E in physical
components from textbook orthonormal-spherical formulas, then converts
to spin basis, then expresses physical-component v in spin components).

## Implementation plan (for next session)

These two BCs are **scalar-valued boundary conditions** at a single
radial boundary (R_inner or R_outer). They fit the existing BC walker
infrastructure for "vector-projection BC" extensions but require new
plumbing because they involve:

1. **Two unknowns simultaneously** (v^- and v^0 for E^{+1,0};
   v^+ and v^0 for E^{-1,0}) — the existing BC walker handles single-
   unknown-component constraints. Need to extend to multi-component
   linear combinations.

2. **A radial derivative ∂_r at the boundary** — already handled by
   the existing BC walker via one-sided stencils built in
   `_lagrange_derivative_weights`.

3. **An angular derivative ∂_θ on the boundary** — already handled
   by the BC walker for D_θ-acting constraints.

4. **Pointwise terms with 1/r and m/sin θ coefficients** — trivial
   pointwise multiplications.

### Suggested API (pending user decision)

Three options, in increasing order of generality:

**Option 1: helper class.**
```python
ImpenetrableStressFree(v, boundary=Boundary.R_OUTER)
```
Emits three internal BCs:
- `Dot(r_hat, v) = 0`  (existing machinery)
- `E^{+1, 0}(v) = 0`  (new BC kind)
- `E^{-1, 0}(v) = 0`  (new BC kind)

**Option 2: explicit BC kind.**
```python
BoundaryCondition(Boundary.R_OUTER, Dot(r_hat, v),     name="v_r")
StressFreeAngularBC(v, sigma=+1, boundary=Boundary.R_OUTER, name="E_p1_0")
StressFreeAngularBC(v, sigma=-1, boundary=Boundary.R_OUTER, name="E_m1_0")
```
User constructs each BC explicitly.

**Option 3: symbolic expression-based.**
```python
E = Grad(v) + Transpose(Grad(v))                 # tensor expression
BoundaryCondition(R_OUTER, Dot(r_hat, v),                    name="v_r")
BoundaryCondition(R_OUTER, Dot(theta_hat, Dot(E, r_hat)),    name="stress_t")
BoundaryCondition(R_OUTER, Dot(phi_hat, Dot(E, r_hat)),      name="stress_p")
```
This requires tensor-expression infrastructure in the IR (Grad-of-vector,
Transpose) and a walker that recognizes the `Dot(unit_vec, Dot(tensor_expr,
r_hat))` pattern. Substantially more work; deferred.

**Recommendation:** Option 1 (helper class). Smallest implementation,
matches the use case exactly. The helper internally constructs the right
BC tasks. If/when the symbolic expression infrastructure becomes valuable
elsewhere (e.g. for non-standard tensor BCs), Option 3 can be added
without removing Option 1.

### BC walker entry point sketch

New BC task kind (in `spdee/ir/tasks.py`):
```python
TASK_BC_STRAIN_ANGULAR_RADIAL = "bc_strain_angular_radial"
```
with metadata `sigma_sign: +1 or -1` indicating which of `E^{+1,0}` or
`E^{-1,0}` is being enforced.

The walker (`spdee/ir/walker.py::_parse_one_bc`) recognizes the helper
class and emits this task. The backend BC dispatch
(`spdee/backends/fd2/compiled_problem.py::_apply_bc_row_replacement`)
implements it by:

1. Identifying the row block in the v unknown's σ=0 slot at the boundary
   (for v_r=0) and the σ=∓ slot (for the angular stress BC affecting v^∓).
2. For each θⱼ on the boundary, constructing the row:
   - Pointwise coefficient on `v^∓(R_b, θⱼ)`: `-1/R_b`
   - Coefficient on `v^0(R_b, θⱼ)`: `±m / (√2 R_b sin θⱼ)`  (+ for σ=+1, - for σ=-1)
   - Radial derivative weights: standard one-sided stencil at i_r=N_r-1
     (R_outer) or i_r=0 (R_inner), applied to `v^∓` block.
   - Angular derivative weights: from the bundle's `D1_θ` operator,
     applied to `v^0` block at row j.
   - RHS = 0.

### Manufactured test for verification

A natural choice: harmonic vector field in a ball satisfying stress-free
at r = R_outer. The simplest such field is **rigid-body rotation**:

```
v = Ω r sin θ ê_φ
```

This has Lap(v) = 0 (harmonic), is impenetrable at any r (v_r = 0
identically), and is stress-free at every r (E[v] = 0 identically — it's
literally a rigid rotation, no deformation). So as a manufactured Poisson
test:
- Equation: Lap(v) = 0
- BCs: impenetrable + stress-free at r=R_outer + anchor on Ω somehow
  (otherwise the field is in the null space of the BC problem)

The anchor problem is genuine: at m=0 ball mode, the constant-Ω rigid
rotation has no other constraint — we'd need to pin Ω somehow. One
option: pick a slightly off-rigid manufactured field with a known
non-zero forcing term that uniquely determines the solution.

The actual test design can be deferred to implementation session.

## Files

- `scratch/derive_strain_bc.py` — derivation of E^{σ,0} formulas from
  physical-component Grad(v) in orthonormal spherical basis.
- `scratch/verify_strain_bc.py` — symbolic verification on three cases:
  rigid-body rotation (BCs = 0), pure-radial flow (BCs = 0 by symmetry),
  pure-θ flow (BCs ≠ 0 in general; = 0 iff f(r) = c·r).
- `DERIVATION_STRAIN_BC.md` — this file.
