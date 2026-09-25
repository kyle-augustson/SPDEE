# Derivation note: |m|=1 vector ball origin anchor

**Date:** 2026-05-14
**Status:** Derivation complete; numerical prototype pending (next session).
**Scope:** Provide a correct anchor for the in-plane Cartesian-vector-at-origin DOF (α) when compiling a ball + vector + |m|=1 system, replacing the current Dirichlet-zero approximation.

## Summary

The earlier "ℓ=1-projected L'Hospital" framing is **misleading**. The |m|=1 case is structurally different from m=0 and cannot use the same L'Hospital strategy. The correct anchor for α is a **first-radial-derivative regularity condition**, not a second-derivative PDE consequence.

Key facts derived this session:

1. **Origin spin patterns** (m=+1, e^{iφ} factored out):
   - v⁰(0, θ) = (α/√2) · sin θ
   - v⁻(0, θ) = (α/2) · (cos θ + 1)
   - v⁺(0, θ) = (α/2) · (cos θ − 1)

2. **Cartesian-smooth projection** (analog of v_z for m=0):
   - v_α(r, θ) = (sin θ /√2) · v⁰ + ((cos θ + 1)/2) · v⁻ + ((cos θ − 1)/2) · v⁺
   - At r=0: v_α = α (θ-independent — as a Cartesian scalar at a point should be).

3. **The standard L'Hospital does NOT apply at m=+1**: for Cartesian-smooth m=+1 fields, ∂²v_α/∂r²(0, θ) ≡ 0. The α DOF is determined by the first radial derivative, not the second:
   - ∂v_α/∂r(0, θ) = −α · sin θ for the leading regularity term.

4. **ℓ=1 projection weights** for m=+1 angular functions: solve V^T w = e_0 where V[j, ℓ−1] = P_ℓ^1(cos θ_j). Verified machine-precision exact for span{P_1^1, ..., P_{N_θ}^1} and super-algebraically convergent for general smooth g(θ) = sin θ · q(cos θ).

## Proposed anchor (for next session's prototype)

For each vector unknown in a ball + vector + |m|=1 system, the i_r=0 row block consists of:

1. **One anchor row** (e.g. at σ=−, j=0, i_r=0; choice of well-conditioned position TBD): the ℓ=1-projected first-radial-derivative regularity condition for v_α:

       Σⱼ w_proj_ℓ1[j] · Σ_k w_lh_d1[k] · v_α(r_k, θⱼ)  =  α_target

   where:
   - `w_lh_d1[k]` is a one-sided first-derivative stencil at r=0 over {r_0, r_1, ..., r_K} (the standard k=4 or 5 node count gives 4th-order accuracy).
   - `w_proj_ℓ1[j]` are the ℓ=1 projection weights solving V^T w = e_0 in the P_ℓ^1 basis on the cell-centered θ grid.
   - v_α(r_k, θⱼ) = (sin θⱼ/√2) v⁰(r_k, θⱼ) + ((cos θⱼ+1)/2) v⁻(r_k, θⱼ) + ((cos θⱼ−1)/2) v⁺(r_k, θⱼ)
   - α_target is what the LHS computes when the field has Cartesian α at the origin. Since ∂v_α/∂r(0, θ) = −α sin θ + (higher-r terms), and the P_1^1 coefficient of −sin θ is 1, the LHS extracts (−α). So the anchor row reads:

         (LHS as written above) = −α

     And since α IS what we're solving for (one of the i_r=0 DOFs in the discrete unknown vector), this becomes an equation relating α to v values at r > 0 grid points. Wait — α isn't a state variable directly; it's *encoded* in v^σ(0, θ) values via the spin patterns. So after substituting the 3N_θ−1 angular-Dirichlet rows that tie v^σ(0, θⱼ) to whatever single DOF we pick as the anchor variable, the anchor row becomes a closed equation in the anchor variable and the v^σ(r_{k>0}, θⱼ) values.

2. **3N_θ − 1 angular Dirichlet rows** tying every v^σ(0, θⱼ) to the chosen anchor variable. Concretely, pick a well-conditioned anchor variable — let's say v⁻(0, θ_0) (since v⁻(0, θ) = α(cos θ + 1)/2 is maximal at small θ where cos θ ≈ 1, so v⁻(0, θ_0) / ((cos θ_0 + 1)/2) extracts α robustly). Then for each (σ, j) ≠ (−, 0):
   - v⁻(0, θⱼ) = v⁻(0, θ_0) · (cos θⱼ + 1)/(cos θ_0 + 1)
   - v⁰(0, θⱼ) = v⁻(0, θ_0) · √2 sin θⱼ / (cos θ_0 + 1)
   - v⁺(0, θⱼ) = v⁻(0, θ_0) · (cos θⱼ − 1)/(cos θ_0 + 1)

3. **Source term:** in contrast to m=0, the source does NOT directly enter the anchor row. The α DOF is determined by global elliptic structure (interior operator + outer BC), not by a local Lap = source identity at r=0. The anchor row has zero RHS (or a small correction TBD).

   This is the most subtle structural point. The m=0 anchor's RHS is the ℓ=0 projection of the source at origin because the L'Hospital identity `3 f_rr(0) = Lap(f)(0)` connects the local DOF (c) to the local source. At |m|=1 there is no analogous connection — the local PDE at r=0 is **automatically satisfied** by regularity (the 1/r singular term cancels because A(t) = −α sin t is forced to be ℓ=1, but the amplitude α is free at the local level). So the anchor row's RHS should be zero. The discrete prototype must verify this empirically.

## Open questions for the prototype

1. **Does the proposed anchor close the system?** I.e., is the resulting 3N_r N_θ × 3N_r N_θ matrix non-singular? Needs numerical check.

2. **What's the RHS of the anchor row?** Default to zero (per the analysis above), but the prototype should test this against a manufactured m=+1 problem with known α and verify convergence.

3. **Anchor variable choice.** v⁻(0, θ_0) suggested above. Could also be v⁰(0, θ_0)/something — but v⁰(0, θ) = (α/√2) sin θ vanishes at θ=0 on a polar-aligned grid (though the cell-centered grid avoids θ=0 exactly), so v⁻ is the more robust choice. Worth a numerical conditioning check.

4. **Stencil width for w_lh_d1.** The m=0 code uses a 5-point one-sided d2 stencil; the d1 analog would use the same node set. Number to pick.

5. **Anchor row placement.** In the m=0 code the anchor is at (σ=0, j=0, i_r=0). For m=+1 the natural placement is at (σ=−, j=0, i_r=0), aligned with the anchor variable choice.

6. **Manufactured solution for testing.** A natural choice:
   - v_full = α · (1 − r²) · (x̂ + iŷ)/√2 · e^{iφ}_phase. At r=0: in-plane Cartesian amplitude α. At r=1: zero. Smooth, simple Lap.
   - Source: compute via SymPy along the same pattern as scratch/check_m2_source.py.
   - Outer BC: zero Dirichlet on all three physical components.
   - Verify the solver recovers α exactly (or close to) and the full v field at 4th-order convergence.

7. **Conjugate m=−1 case.** Symmetry: m=−1 is the complex conjugate of m=+1. Spin patterns flip sign on cos θ ± 1? Need to confirm by re-deriving for m=−1, but expectation is the same algebra structure with signs flipped on the v⁻ ↔ v⁺ roles.

## Files produced this session

All in `scratch/`:

- `derive_m1_origin_step1.py` — Origin spin patterns (v^σ(0, θ) for α DOF).
- `derive_m1_origin_step2.py` — v_α Cartesian-smooth projection.
- `derive_m1_origin_step3.py` — Verification on three non-trivial Cartesian-smooth fields.
- `derive_m1_origin_step4_crosscheck.py` — Cross-check with m=0 v_z formula.
- `derive_m1_lhospital.py` — Demonstration that the standard L'Hospital does not apply at m=+1.
- `derive_m1_ell1_weights.py` — ℓ=1 projection weights, with five test cases including super-algebraic convergence.

## Path 1 vs Path 2 reassessment

Now that the derivation is concrete, the implementation choice between Path 1 (in-place anchor in compiled_problem.py) and Path 2 (explicit α DOF in state layout) can be re-evaluated:

- **Path 1 still looks reasonable.** The anchor variable v⁻(0, θ_0) is "implicit α" — it lives in the existing state layout at a specific grid point, and the angular Dirichlet rows tie everything else to it. No state-layout surgery required. The prototype should attempt this first.

- **Path 2 would be cleaner conceptually** but heavier on plumbing. If Path 1 hits conditioning trouble or doesn't extend naturally (e.g., to tensor origin DOFs), Path 2 becomes the fallback.

Recommendation for next session: **prototype Path 1 first**. If it works with reasonable convergence, ship it. If not, the derivation here transfers to Path 2 directly (same math, different storage).

## Implementation-ready helper (drop-in for compiled_problem.py)

```python
from scipy.special import lpmv

def _ball_origin_ell1_m1_projection_weights(theta: np.ndarray) -> np.ndarray:
    """Return weights w[j] such that
        sum_j w[j] g(θ_j)  ≈  P_1^1-coefficient of g
    where g(θ) is the angular part of an m=+1 Cartesian-smooth scalar
    (with the e^{iφ} factor pulled out).

    Construction
    ------------
    V[j, ℓ-1] = P_ℓ^1(cos θ_j), ℓ = 1, ..., N_θ.
    Solve V^T w = e_0.

    Exact for any function in span{P_1^1, ..., P_{N_θ}^1};
    super-algebraically convergent for general Cartesian-smooth m=+1
    angular functions. Verified in scratch/derive_m1_ell1_weights.py.
    """
    N = len(theta)
    cos_theta = np.cos(theta)
    V = np.zeros((N, N))
    for ell in range(1, N + 1):
        V[:, ell - 1] = lpmv(1, ell, cos_theta)
    e0 = np.zeros(N)
    e0[0] = 1.0
    return np.linalg.solve(V.T, e0)
```
