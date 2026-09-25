# Derivation: sparse (A_op, B_op) for Div(strain v) in V19 spin basis

## Goal

Replace the dispatcher's `first_derivative_operators` (which materializes
`D = A⁻¹·B` as dense) with a builder that returns `(A_op, B_op)` both
sparse banded, where
```
A_op · [r²·(Div strain v)^σ_out] = B_op · v
```
for each output spin σ_out ∈ {-1, 0, +1}.

The mass A_op is identical for all three output blocks (it's the
discretization's intrinsic mass for r²-formulated 2nd-order operators):
```
A_op_block = A_2_r · diag(r²) · A_2_θ   (Kronecker-lifted)
```
And the full A_op is `block_diag(A_op_block, A_op_block, A_op_block)`,
same as lap_vector.

## Starting point: spin-basis formula for r²·(Div strain v)^σ

Substituting Stage 1 (U_in_v) into Stage 2 (per_sigma_expr_in_U)
analytically and expanding:

### σ_out = ±1
```
r²·(Div strain v)^{±} =
    r²·v^{±},rr  + 2r·v^{±},r                                  (T1)
  + (3/2)·v^{±},θθ + (3/(2 tan θ))·v^{±},θ
        − (3/2)·(1 + cot²θ)·v^{±}                              (T2)
  + (1/2)·v^{∓},θθ + (1/(2 tan θ))·v^{∓},θ
        − (1/2)·(1 + cot²θ)·v^{∓}                              (T3, ∓-cross)
  + (√2/2)·r·v^0,rθ
  + 2√2·v^0,θ                                                  (T4, 0-cross)
```
Note: the +(σ=+1) and −(σ=−1) cases differ only by which sigma is "self"
and which is "cross"; the formula is symmetric under m↔p with self-cross
roles swapped. The √2·r·v^0,rθ term is the same for both σ_out=±1.

### σ_out = 0
```
r²·(Div strain v)^0 =
    2·r²·v^0,rr + 4r·v^0,r                                     (S1)
  + v^0,θθ + cot(θ)·v^0,θ                                      (S2)
  − 4·v^0                                                      (S3)
  + (√2/2)·[r·v^{−},rθ + r·v^{+},rθ]                           (S4, ±-cross)
  + (√2/2)·cot(θ)·[r·v^{−},r + r·v^{+},r]                      (S5)
  − (3√2/2)·[v^{−},θ + v^{+},θ]                                (S6)
  − (3√2/(2 tan θ))·[v^{−} + v^{+}]                            (S7)
```

## Discretization strategy

Same pattern as lap_scalar / lap_vector:

1. **Multiply by r through product rule** to remove explicit r·∂_r pieces
   wherever possible:
   - σ=±1: `r²·v^{±},rr + 2r·v^{±},r = r·(r·v^{±})_rr`
   - σ=0: `2·r²·v^0,rr + 4r·v^0,r = 2·r·(r·v^0)_rr`
   These get discretized via `A_2_r · (r·v),rr = B_2_r · (r·v)`, i.e.,
   the radial compact-2nd-derivative stencil acts on `r·v` directly.

2. **Compact 2nd-derivative for θθ** terms: standard `A_2_θ · v,θθ = B_2_θ · v`.

3. **Standard (A=I) 1st-derivative for variable-coefficient θ terms**:
   any term with `cot(θ)·∂_θ` or just `∂_θ` multiplied by something
   θ-dependent uses `B_1_θ_std` so the variable coefficient multiplies
   on the left without inversion.

4. **Mixed rθ-derivative `r·v^σ,rθ`**: write as `∂_θ(r·v^σ,r)`.
   - Inner: `r·v^σ,r` discretized via compact `A_1_r · v^σ,r = B_1_r · v^σ`
     scaled by `diag(r)`: `(A_1_r) · (r·v^σ,r) = diag(r) · B_1_r · v^σ`
     [the lift of this 1D relation needs `r` factor in the A side too...
     actually: `r·v^σ,r` is just `diag(r) · ∂_r v^σ`. With A_1_r·∂_r v = B_1_r·v,
     we have `r·v^σ,r = diag(r) · A_1_r⁻¹ · B_1_r · v^σ` — but we don't
     want the inverse. Better: keep `r·v^σ,r` as an auxiliary and apply
     ∂_θ-std on the outside. The whole term becomes
     `B_1_θ_std · diag(r) · (A_1_r⁻¹ · B_1_r) · v^σ` — still has the
     inverse.]
   
   ALTERNATIVE: introduce a NEW radial stencil that handles `r·∂_r v`
   directly without an inverse. Note: `r·∂_r v = ∂_r(r·v) − v`. Then
   `A_1_r · ∂_r(r·v) = B_1_r · (r·v)`, so `A_1_r · (r·v^σ,r + v^σ) = B_1_r · (r·v^σ)`,
   i.e., `A_1_r · (r·v^σ,r) = B_1_r · diag(r) · v^σ − A_1_r · v^σ`.
   In Kronecker form lifted to 2D:
   ```
   A_1_r · [r·∂_r v^σ] = (B_1_r · diag(r) − A_1_r) · v^σ
   ```
   This is sparse banded on both sides. The full A_op already has
   `A_1_r` as a factor in the radial axis (it's a tridiagonal compact
   1st-deriv stencil). The mass for mixed-derivative terms then needs
   `A_1_r ⊗ I_θ` rather than `A_2_r ⊗ A_2_θ`.

   **PROBLEM**: a different mass for different terms means we can't
   share a single A_op. The lap_scalar strategy of using A_2_r ⊗ A_2_θ
   as the universal mass requires that ALL terms be expressible as
   `(A_2_r ⊗ A_2_θ) · (something on v) = (sparse on v)`. For the mixed
   rθ term, we'd need
   `(A_2_r ⊗ A_2_θ) · (r·v^σ,rθ) = (A_2_r · ?·) ⊗ (A_2_θ · ∂_θ) · v^σ` — 
   but A_2_r · (r · ∂_r) doesn't factor cleanly.

## Decision point

Option A: Use the SAME mass A_op = A_2_r diag(r²) A_2_θ for ALL terms.
For the mixed rθ term, accept that the relation
```
(A_2_r diag(r²) A_2_θ) · (r·v^σ,rθ) = ?
```
can't be expressed with just compact stencils. Fall back to STANDARD
d_r for the radial part of the mixed term (A_1_r_std = I), so
```
(A_2_r diag(r²) A_2_θ) · (r·v^σ,rθ) = (A_2_r diag(r²)) · diag(r)·B_1_r_std · A_2_θ · ∂_θ v^σ
                                    = (A_2_r diag(r³) B_1_r_std) · A_2_θ · ∂_θ v^σ
```
Wait — but A_2_θ · ∂_θ also has an inversion problem. Use std d_θ too:
```
(A_2_r diag(r²) A_2_θ) · (r·v^σ,rθ) = (A_2_r diag(r³) B_1_r_std) ⊗ (A_2_θ · B_1_θ_std)
```
Each piece is sparse banded. The (A_2_θ · B_1_θ_std) part needs A_2_θ
to multiply B_1_θ_std on the left — A_2_θ is tridiagonal (5 nonzeros
per row in the lifted form), B_1_θ_std has a 5-point stencil, the
product has bandwidth ~9. Still sparse and banded.

This is the cleanest approach: use STANDARD stencils for any term that
involves a 1st-derivative (whether r, θ, or mixed), keeping the
2nd-derivative pieces with compact. This is a slight accuracy hit
(standard d1 is 4th-order vs compact d1 is 4th-order — same order!
just different coefficients), but the SPARSITY is preserved.

WAIT — standard d1 4th-order has a 5-point stencil; that's wider than
compact d1's tridiagonal A_1 + tridiagonal B_1 = total 5 points effective
support. So standard d1 has the same accuracy as compact d1 in terms
of formal order. The "compact" advantage is implicit; for our purposes
where we want sparse rather than implicit, using standard d1 throughout
is the right call.

Option B: Different mass for the mixed-rθ piece. Complicates the
dispatcher significantly. Not worth it.

## Final plan: Use STANDARD d1 throughout, COMPACT d2 throughout.

Mass matrix:
```
A_op = A_2_r diag(r²) ⊗ A_2_θ
```

For each term in r²·(Div strain v)^σ:

**(rv),rr terms** [σ=0: 2·r·(r·v^0)_rr; σ=±1: r·(r·v^{±})_rr]
```
A_2_r · (r·v),rr = B_2_r · (r·v)
A_op · 2·r·(r·v^0)_rr = (B_2_r · diag(2r) · diag(r)) ⊗ A_2_θ · v^0
                      = (B_2_r · diag(2r²)) ⊗ A_2_θ · v^0
```
σ=±1 similarly with diag(r²) instead of diag(2r²).

**v,θθ terms** [σ=0: v^0,θθ; σ=±1: (3/2)·v^σ,θθ + (1/2)·v^{∓},θθ]
```
A_2_θ · v,θθ = B_2_θ · v
A_op · v^0,θθ = A_2_r · diag(r²) · A_2_θ · v^0,θθ
              = A_2_r · diag(r²) · B_2_θ · v^0
              (factoring: A_2_r diag(r²) ⊗ B_2_θ acting on v^0)
```

**cot(θ)·v,θ terms** [variable-coef 1st-derivative in θ]
Use B_1_θ_std (A_1_θ_std = I).
```
A_op · (cot θ · v,θ) = A_2_r diag(r²) ⊗ (A_2_θ · diag(cot θ) · B_1_θ_std) · v
```

**(1/sin²θ + 1)·v terms** [diagonal-in-v with θ-dependent coefficient]
```
A_op · (c(θ) v) = A_2_r diag(r²) ⊗ (A_2_θ diag(c)) · v
```

**v,θ terms (no cot)** [σ=0: −(3√2/2)·v^{∓},θ]
```
A_op · v,θ = A_2_r diag(r²) ⊗ (A_2_θ · B_1_θ_std) · v
```

**r·v,rθ mixed term** [σ=0: (√2/2)·(r·v^{∓},rθ); σ=±1: (√2/2)·r·v^0,rθ]
Write as `∂_θ(r·v,r) − v,θ` ... actually no, just use std d_r and std d_θ:
With A_1_r_std = I and A_1_θ_std = I, `B_1_r_std · v = v,r` and
`B_1_θ_std · v = v,θ`. Then
```
A_op · (r · v,rθ) = A_op · (r · B_1_r_std · B_1_θ_std · v)
                  = (A_2_r diag(r²) · diag(r) · B_1_r_std) ⊗ (A_2_θ · B_1_θ_std) · v
                  = (A_2_r diag(r³) B_1_r_std) ⊗ (A_2_θ B_1_θ_std) · v
```
Sparse banded. ✓

**cot(θ)·r·v,r term** [σ=0: (√2/2)·cot(θ)·r·v^{∓},r]
```
A_op · (cot θ · r · v,r) = (A_2_r diag(r²) · diag(r) · B_1_r_std) ⊗ (A_2_θ · diag(cot θ)) · v
                         = (A_2_r diag(r³) B_1_r_std) ⊗ (A_2_θ diag(cot θ)) · v
```

## Sparsity analysis

Every piece of B_op has the form `(radial banded) ⊗ (angular banded)`,
where each axis-part is bandwidth ≤ ~7-9 (compact-2nd-deriv is
tridiagonal product = bandwidth 5; standard-1st-deriv is 5-point so
A_2_r · B_1_r_std has bandwidth ≤ 7).

In the lifted 2D matrix, each row of B_op has ≤ (bandwidth_r × bandwidth_θ)
nonzeros, which is roughly 7 × 7 ≈ 49 entries per row. For N_r = N_θ = 96,
that's 49 × N_r·N_θ ≈ 450k nnz per spin block. Three spin blocks =
~1.3M nnz total. Compare to the current 44%-dense matrix at N=96 which
would be ~340M nnz. **~250× sparser.**

## Ball-mode parity considerations

Same as lap_vector_ball_m0: the radial compact stencils need parity-folded
boundary closures at r=0. Use bundle.parity_d2_r(parity=+1) etc.
For m=0 with strain, parity at r=0 follows the same rules as Lap:
v^0 even (parity +1), v^± odd (parity −1). The standard d1 in r is
unaffected by ball boundary (it's an explicit stencil with no inverse).

## Origin regularity

Same factor-4 treatment as before. Computed once by walker via
`_compute_origin_factor_for_div` and stored in metadata.

## What changes in the dispatcher

The dispatcher currently:
  1. Walks Stage 1 to lift U from v (using first_derivative_operators).
  2. Walks Stage 2 to compose Div(U) (using first_derivative_operators).
  3. Builds dense ~half-N² operator.

The new dispatcher:
  1. Calls `div_strain_vector(bundle, m)` which returns (A_op, B_op).
  2. Accumulates B_op into A_total like _dispatch_lap_vector does.
  3. (A_op handled by _determine_equation_mass.)

`_determine_equation_mass` returns A_op for the Div task, which is the
same as the lap_vector mass at present (A_2_r diag(r) ⊗ A_2_θ for shell,
appropriate ball form). For Div(strain) we need A_2_r diag(r²) ⊗ A_2_θ
since the formula is r²-multiplied.

Note: the current `_determine_equation_mass` for Div returns
`r2_vector_mass(bundle)` which is just diag(r²) — that's NOT the same
as A_2_r diag(r²) A_2_θ. We need to upgrade it.
